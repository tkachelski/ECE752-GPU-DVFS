#include "sim/gpu_dvfs_handler.hh"
#include "base/trace.hh"
#include "debug/GpuDVFS.hh"
#include "sim/stat_control.hh"

// definition of Shader for the dynamic_cast
#include "gpu-compute/shader.hh"
#include "gpu-compute/compute_unit.hh"
#include <cmath>

namespace gem5
{

// ----------------------------------------------------------------------
// Constructor
// ----------------------------------------------------------------------
GpuDVFSHandler::GpuDVFSHandler(const Params &p)
    : SimObject(p),
      sysClkDomain(p.sys_clk_domain),
      enableHandler(p.enable),
      _transLatency(p.transition_latency),
      threshold1(p.threshold1),
      threshold2(p.threshold2),
      printToScreen(p.printToScreen),
      // Cast the generic SimObject pointer from Python to a Shader pointer
      gpuShader(dynamic_cast<Shader*>(p.shader)),
      // Initialize the decisionEvent to call 'runDecisionLoop' when triggered
      decisionEvent([this]{ runDecisionLoop(); }, name())
{
    // Populate the map of domains provided in the Python config
    for (auto *d : p.domains) {
        domains[d->domainID()] = d;
    }
    // DEBUG PRINT 1: Constructor
    inform("GPU_DVFS: Handler Created. controlling %d domains.", domains.size());
}

// ----------------------------------------------------------------------
// Startup
// ----------------------------------------------------------------------
void GpuDVFSHandler::startup()
{
    // DEBUG PRINT 2: Startup
    inform("GPU_DVFS: Startup called.");

    if (enableHandler) {
        // Wait 5ms for OS boot before first poll
        inform("GPU_DVFS: Enabled! Scheduling first check in 5ms.");
        schedule(decisionEvent, curTick() + 5000000000); 
    } else {                                
        inform("GPU_DVFS: Handler is DISABLED in params. No logic will run.");
    }
}

// ----------------------------------------------------------------------
// Helper: Find a domain
// ----------------------------------------------------------------------
SrcClockDomain *
GpuDVFSHandler::findDomain(DomainID domain_id) const
{
    auto it = domains.find(domain_id);
    if (it == domains.end()) return nullptr;
    return it->second;
}

// ----------------------------------------------------------------------
// GLOBAL WAVEFRONT SCANNER (Data Collection Phase)
// ----------------------------------------------------------------------
std::map<Addr, int> GpuDVFSHandler::scanGlobalWavefrontState()
{
    std::map<Addr, int> pcHistogram;

    if (!gpuShader) {
        inform("GPU_DVFS ERROR: gpuShader pointer is NULL!");
        return pcHistogram;
    }

    int totalWavesSeen = 0;

    // Iterate over ALL Compute Units
    for (auto *cu : gpuShader->cuList) {
        // Iterate over ALL SIMDs
        for (const auto &simd_waves : cu->wfList) {
            // Iterate over ALL Wavefronts
            for (auto *wave : simd_waves) {
                totalWavesSeen++; 
                if (wave->getStatus() != Wavefront::S_STOPPED) {
                    // Add to histogram: This counts how many waves are at this specific PC
                    pcHistogram[wave->pc()]++;
                }
            }
        }
    }

    return pcHistogram;
}

// ----------------------------------------------------------------------
// CHECK IF GPU IS RUNNING 
// ----------------------------------------------------------------------
int GpuDVFSHandler::checkIfGPUIsRunning()
{
    static int detected = 0;
    //chaanged use of IPC stats to Check physical wavefront status directly.
    if (!gpuShader) return 0;

    if(!detected){
        for (auto *cu : gpuShader->cuList) {
            for (const auto &simd_waves : cu->wfList) {
                for (auto *wave : simd_waves) {
                    // If any wave is NOT stopped, the GPU is running
                    double ipc = cu->stats.ipc.total();
                    //if(!std::isnan(ipc) && ipc > 0){
                    if (wave->getStatus() != Wavefront::S_STOPPED) {
                       return 1;
                    //}
                    }
                }
            }
        }
    }
    return 0;
}


int GpuDVFSHandler::computeUnitSensitivity()
{
    static double prevInstrExecuted[40] = {0};
    // Iterate over ALL Compute Units
    int index = 0;
    for (auto *cu : gpuShader->cuList) {
        // Iterate over ALL SIMDs
        double ipc = cu->stats.ipc.total();
        if(!std::isnan(ipc) && ipc > 0){
            double instr = cu->stats.numInstrExecuted.total();
            double deltaInstr = instr - prevInstrExecuted[index];
            if(deltaInstr < 0) deltaInstr = instr;
            sensitivity[index] = deltaInstr / (cu->frequency()/1000000000);
            //sensitivity[index] = cu->frequency() / newInstr;
            
            //inform("GPU_DVFS: CU %d inst: %d, recent: %d, freq: %d, sensitivity is %f", index, instr, newInstr, cu->frequency(), sensitivity[index]);
            prevInstrExecuted[index] = instr;
        }
        index++;
    }
    return 0;
}


int GpuDVFSHandler::dumpImportantStatsToConsole()
{
    static double prevInstTotal[40] = {0};
    static double prevNumCycles[40] = {0};
    static double initTime = 0;
    static double edpTotal = 0.0;
    static double ed2pTotal = 0.0;
    // Iterate over ALL Compute Units
    int index = 0;
    for (auto *cu : gpuShader->cuList) {
        // Iterate over ALL SIMDs
        double ipc = cu->stats.ipc.total();
        if(!std::isnan(ipc) && ipc > 0){
            if(initTime == 0){
                initTime = curTick();
            }
            double instr = cu->stats.numInstrExecuted.total();
            double deltaInstr = instr - prevInstTotal[index];
            double numCycles = cu->stats.totalCycles.total();
            double deltaNumCycles = numCycles - prevNumCycles[index];
            if(deltaInstr < 0) deltaInstr = instr;
            if(deltaNumCycles < 0) deltaNumCycles = numCycles;
            double deltaIPC = deltaInstr / deltaNumCycles;
            double edp = cu->voltage() * cu->voltage() * cu->frequency() * deltaInstr;
            double ed2p = edp * deltaInstr;
            edpTotal += edp; 
            ed2pTotal += ed2p; 
            inform("GPU_DVFS_STATS: CU: %d, perfLevel: %d clock: %d, Cycles: %d, IPC: %f, IPC_delta: %f, CPI: %f, CPI_delta: %f, Frequency: %d, Voltage: %f, EDP: %f, ED2P: %f, EDP_Total: %f, ED2P_Total: %f, Sensitivity: %f"
               , index
               , domains.begin()->second->perfLevel()
               , curTick() - initTime
               , cu->stats.totalCycles.total()
               , cu->stats.ipc.total()
               , deltaIPC
               , (cu->stats.ipc.total() > 0) ? (1.0 / cu->stats.ipc.total()) : 0
               , (deltaIPC > 0) ? (1.0 / deltaIPC) : 0
               , cu->frequency()
               , cu->voltage()
               , edp
               , ed2p
               , edpTotal
               , ed2pTotal
               , sensitivity[index]
            );
            prevInstTotal[index] = instr;
            prevNumCycles[index] = numCycles;
        }
        index++;
    }
    return 0;
}

double  GpuDVFSHandler::sensitivityAverage(){
    double total = 0.0;
    int count = 0;
    for(int i = 0; i < 40; i++){
        if(sensitivity[i] > 0){
            total += sensitivity[i];
            count++;
        }
    }
    if(count == 0) return 0.0;
    return total / count;
}

// ----------------------------------------------------------------------
// THE PCSTALL GOVERNOR LOGIC (Decision Phase)
// ----------------------------------------------------------------------
void GpuDVFSHandler::runDecisionLoop()
{
    static bool hasPrintedRunning = false;
    static int idleHeartbeat = 0;
    static double maxSensitivity = 0.0;

    int isRunning = checkIfGPUIsRunning();

    if(!isRunning) {
        // Print a heartbeat every 10,000 checks so we know sim isn't frozen
        if (++idleHeartbeat % 10000 == 0) {
           inform("GPU_DVFS: Waiting for GPU kernel... (Check #%d)", idleHeartbeat);
        }

        // Relax polling to 100us (100,000,000 ticks) to reduce overhead
        // while the OS is booting or app is initializing.
        schedule(decisionEvent, curTick() + 100000000);
        return;
    }
    else {
        // Reset heartbeat logic once running
        idleHeartbeat = 0;
        if(!hasPrintedRunning){
           statistics::reset();
           inform("GPU_DVFS: GPU KERNEL DETECTED! DVFS Active.");
           hasPrintedRunning = true;
        }
    }

    if (domains.empty()) {
        inform("GPU_DVFS ERROR: No domains registered!");
        return;
    }
    computeUnitSensitivity();
    double average = sensitivityAverage();
    if(average > maxSensitivity){
        maxSensitivity = average;
    }
    inform("GPU_DVFS: Average CU Sensitivity: %f", average);
    if(printToScreen)
        dumpImportantStatsToConsole();

    // Grab the first available domain (since we only have one GPU domain)
    auto it = domains.begin();
    SrcClockDomain *domain = it->second;
    DomainID targetDomain = it->first; 

    // 1. GATHER PHASE
    std::map<Addr, int> pcMap = scanGlobalWavefrontState();

    // Poll period: 10us (10,000,000 ticks) when active
    Tick nextPollTick = 100000;

    // Case 1: GPU is IDLE (Map is empty)
    if (pcMap.empty()) {
        // GPU became idle mid-execution
        schedule(decisionEvent, curTick() + nextPollTick);
        return;
    }


    // 2. ANALYZE PHASE: Calculate PC Concentration
    int maxWavesAtOnePC = 0;
    int totalActiveWaves = 0;
    Addr dominantPC = 0;

    for (auto const& [pc, count] : pcMap) {
        totalActiveWaves += count;
        if (count > maxWavesAtOnePC) {
            maxWavesAtOnePC = count;
            dominantPC = pc;
        }
    }

    // "Concentration" metric: 0.0 to 1.0
    // High Concentration implies waves are synchronized at a bottleneck (Stall).
    // Low Concentration implies waves are executing freely (Compute).
    double concentration = 0.0;
    static double prevConcentration = 0;;
    if (totalActiveWaves > 0) {
        concentration = (double)maxWavesAtOnePC / totalActiveWaves;
    }

    // 3. DECISION PHASE
    PerfLevel currentLevel = domain->perfLevel();
    PerfLevel desiredLevel = currentLevel;

    // > 50% Concentration -> STALL -> Low Freq (Level 2)
    // < 50% Concentration -> COMPUTE -> High Freq (Level 0)
    //if (concentration > 0.66) {
    //    desiredLevel = 2; // Low Perf
    //}else if (concentration > 0.33) {
    //    desiredLevel = 1; // Med Perf
    //} 
    //else {
    //    desiredLevel = 0; // Max Perf
    //}
   // desiredLevel = 2;

   static int count1 = 0;
   static int count2 = 0;
   static int count3 = 0;


    if (average/maxSensitivity > .66) {
        //desiredLevel = 0; // Max Perf
        count1++;
        count2 = count2 > 0 ? count2 - 1 : 0;
        count3 = count3 > 0 ? count3 - 1 : 0;
    } else if(average/maxSensitivity > .33){
        count2++;
        count1 = count1 > 0 ? count1 - 1 : 0;
        count3 = count3 > 0 ? count3 - 1 : 0;
        //desiredLevel = 1; // Med Perf
    }else {
        count1 = count1 > 0 ? count1 - 1 : 0;
        count2 = count2 > 0 ? count2 - 1 : 0;
        count3++;
        //desiredLevel = 2; // Low Perf
    }

    if(count1 >=2){
        desiredLevel = 0; // Max Perf
    }else if(count2 >=2){
        desiredLevel = 1; // Med Perf
    }else if(count3 >=2){
        desiredLevel = 2; // Low Perf
    }
    if(desiredLevel != currentLevel){
        count1 = 0;
        count2 = 0;
        count3 = 0;
    }

    maxSensitivity *= 0.95; // Decay over time

    // 4. ACTUATION PHASE
    if (desiredLevel != currentLevel) {
        inform("GPU_DVFS: Update %d -> %d | ActiveWaves: %d | Concentration: %.2f",
               currentLevel, desiredLevel, totalActiveWaves, concentration);

        auto *e = new UpdateEvent();
        e->handler = this;
        e->domainIDToSet = targetDomain;
        e->perfLevelToSet = desiredLevel;
        e->updatePerfLevel();
    }

    // Schedule next check
    schedule(decisionEvent, curTick() + nextPollTick);
    
}


// ----------------------------------------------------------------------
// Actuation Event
// ----------------------------------------------------------------------
void GpuDVFSHandler::UpdateEvent::updatePerfLevel()
{
    // This dumping is what creates the multiple blocks in stats.txt
    //statistics::dump();
    //statistics::reset();

    auto d = handler->findDomain(domainIDToSet);
    d->perfLevel(perfLevelToSet);
} 

} // namespace gem5

