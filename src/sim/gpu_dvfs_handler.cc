#include "sim/gpu_dvfs_handler.hh"
#include "base/trace.hh"
#include "debug/GpuDVFS.hh"
#include "sim/stat_control.hh"
#include "gpu-compute/shader.hh"
#include "gpu-compute/compute_unit.hh"
#include <cmath>
#include <limits>
#include "sim/core.hh"

namespace gem5
{

// ----------------------------------------------------------------------
// CONSTANTS FOR TUNING
// ----------------------------------------------------------------------
// Target IPC to consider a wave "Compute Bound". 
// 0.08 is a reasonable target for a healthy vector workload.
const double TARGET_WAVE_IPC = 0.08; 

// Base Performance for Memory-Bound code (Sensitivity = 0).
// This prevents "Divide by Zero" in ED2P and represents DRAM throughput.
const double BASE_PERF_MEM = 0.2; 

GpuDVFSHandler::GpuDVFSHandler(const Params &p)
    : SimObject(p),
      sysClkDomain(p.sys_clk_domain),
      enableHandler(p.enable),
      _transLatency(p.transition_latency),
      pollingInterval(p.polling_interval),
      gpuShader(dynamic_cast<Shader *>(p.shader)),
      decisionEvent([this]{ runDecisionLoop(); }, name())
{
    int cuId = 0;
    for (auto *cu : gpuShader->cuList) {
        DomainID did = cuId;
        if (!p.domains.empty()) {
            domains[did] = p.domains[cuId % p.domains.size()];
        }
        cuToDomain[cu] = did;
        cuIdMap[cu] = cuId++;
    }
}

void GpuDVFSHandler::startup()
{
    if (enableHandler) {
        // Init table to 0.5 (Neutral) instead of 0.0
        for (int i = 0; i < TABLE_SIZE; i++) sensitivityTable[i] = 0.5;

        wavefrontLastPC.clear();
        wfCreationTick.clear();
        lastWfInstCount.clear();
        lastWfSchCycles.clear();

        schedule(decisionEvent, curTick() + 5000000000); 
    }
}

SrcClockDomain *
GpuDVFSHandler::findDomain(DomainID domain_id) const
{
    auto it = domains.find(domain_id);
    return (it != domains.end()) ? it->second : nullptr;
}

int GpuDVFSHandler::checkIfGPUIsRunning()
{
    if (!gpuShader) return 0;
    for (auto *cu : gpuShader->cuList) {
        for (const auto &simd_waves : cu->wfList) {
            for (auto *wave : simd_waves) {
                if (wave->getStatus() != Wavefront::S_STOPPED) return 1;
            }
        }
    }
    return 0;
}

// --------------------------------------------------------------------------
// 1. SENSITIVITY MODEL 
// --------------------------------------------------------------------------
double GpuDVFSHandler::computeSensitivity(double deltaInsts, double deltaSchCycles)
{
    if (deltaSchCycles <= 0) return 0.0;

    // 1. Only use Activity (IPC). Removed stall_ratio to prevent inversion bug.
    double ipc = deltaInsts / deltaSchCycles;
    double activity_score = ipc / TARGET_WAVE_IPC;
    // Clamp to reasonable range [0, 2.0] to prevent outliers
    if (activity_score > 2.0) activity_score = 2.0;

    // // 2. Structural Stalls (Frequency Sensitive)
    // double stall_ratio = deltaSchStalls / deltaSchCycles;
   
    // 3. Add them up
    // double S = activity_score + stall_ratio;
    double S = activity_score;
    return S; 
}

void GpuDVFSHandler::updateSensitivityTable(Addr pc, double s_measured)
{
    int idx = getIndex(pc);
    // EMA Update: 75% History, 25% New
    sensitivityTable[idx] = (sensitivityTable[idx] * 0.75) + (s_measured * 0.25);
}

// --------------------------------------------------------------------------
// 2. PERFORMANCE & POWER MODELS
// --------------------------------------------------------------------------
double GpuDVFSHandler::predictPerf(double S, double fMHz, double fNomMHz)
{
    // If S=1, Perf scales linearly with Freq.
    // If S=0, Perf is fixed at BASE_PERF_MEM.
    double fNorm = fMHz / fNomMHz;
    return BASE_PERF_MEM + (S * fNorm);
}

double GpuDVFSHandler::computePower(double fMHz, double v)
{
    // Power = C * V^2 * f
    double dynamic = C_DYNAMIC * v * v * (fMHz / 1000.0) * A_ACTIVITY; 
    return dynamic;
}

double GpuDVFSHandler::computeED2P(double perf, double power)
{
    if (perf <= 1e-6) return 1e15; // Penalty for zero perf
    // Cost = Power / Perf^3
    return power / (perf * perf * perf);
}

// --------------------------------------------------------------------------
// 3. DECISION LOGIC
// --------------------------------------------------------------------------
GpuDVFSHandler::PerfLevel GpuDVFSHandler::chooseBestLevel(double cuSumS, int cuID)
{
    double minCost = std::numeric_limits<double>::max();
    PerfLevel bestLevel = 0;

    for (int lvl = 0; lvl < NUM_LEVELS; ++lvl) {
        double f = freqsMHz[lvl];
        double v = volts[lvl];
        
        double perf = predictPerf(cuSumS, f);
        double power = computePower(f, v);
        double cost = computeED2P(perf, power); // Using ED2P
        
        if (cost < minCost) {
            minCost = cost;
            bestLevel = lvl;
        }
    }
    
     // DEBUG: Print the math for CU 0
    inform("DECISION CU%d: Sensitivty per CU=%.4f |-> Pick Level %d ", 
       cuID, cuSumS, bestLevel);

    return bestLevel;
}

// --------------------------------------------------------------------------
// MAIN LOOP (FIXED PC TRACKING)
// --------------------------------------------------------------------------
void GpuDVFSHandler::runDecisionLoop()
{
    if (!checkIfGPUIsRunning()) {
        schedule(decisionEvent, curTick() + pollingInterval * 10);
        return;
    }

    currentCuSensitivity.clear();

    for (auto *cu : gpuShader->cuList) {
        double cuPredictedSumS = 0.0;
        double activeWaveCount = 0.0;
        
        for (const auto &simd_waves : cu->wfList) {
            for (auto *wf : simd_waves) {
                // Initialize new wavefronts
                if (wfCreationTick.find(wf) == wfCreationTick.end()) {
                    wfCreationTick[wf] = curTick();
                    lastWfInstCount[wf] = wf->stats.numInstrExecuted.total();
                    lastWfSchCycles[wf] = wf->stats.schCycles.total();
                    // Don't have a previous PC yet, so just store current and skip update
                    wavefrontLastPC[wf] = wf->pc();
                    continue; 
                }

                // 1. Measure Deltas
                double currentInsts = wf->stats.numInstrExecuted.total();
                double deltaInsts = currentInsts - lastWfInstCount[wf];
                lastWfInstCount[wf] = currentInsts;

                double currentSchCycles = wf->stats.schCycles.total();
                double deltaSchCycles = currentSchCycles - lastWfSchCycles[wf];
                lastWfSchCycles[wf] = currentSchCycles;

                // 2. Compute Sensitivity (Activity Score)
                double S_measured = computeSensitivity(deltaInsts, deltaSchCycles);
                
                // 3. Update Table using PREVIOUS PC
                // verify the WF is active (isRunning) or did work
                bool isRunning = (wf->getStatus() != Wavefront::S_STOPPED);
                bool didWork = (deltaInsts > 0);

                if (wavefrontLastPC.count(wf)) {
                    Addr prevPC = wavefrontLastPC[wf];
                    //EMA
                    updateSensitivityTable(prevPC, S_measured);
                }

                // 4. Predict using CURRENT PC
                if (isRunning || didWork) {
                    Addr currentPC = wf->pc();
                    int idx = getIndex(currentPC);
                    
                    cuPredictedSumS += sensitivityTable[idx];
                    //track active waves
                    activeWaveCount++; 
                    
                    // Update state for NEXT epoch
                    wavefrontLastPC[wf] = currentPC;
                } else {
                    // Wavefront stopped, cleanup
                    wavefrontLastPC.erase(wf);
                }
            }
        }
     
        // Sum- Average the sensitivity
        if (activeWaveCount > 0) {
            cuPredictedSumS /= activeWaveCount;
        }
        currentCuSensitivity[cu] = cuPredictedSumS; 
       
        int cuID = cuIdMap[cu];
        PerfLevel desiredLevel = chooseBestLevel(cuPredictedSumS, cuID); 

        DomainID did = cuToDomain[cu];
        SrcClockDomain *domain = findDomain(did);
        if (domain && desiredLevel != domain->perfLevel()) {
            auto *e = new UpdateEvent();
            e->handler = this;
            e->domainIDToSet = did;
            e->perfLevelToSet = desiredLevel;
            schedule(e, curTick() + _transLatency);
        }
    }

    dumpImportantStatsToConsole();
    schedule(decisionEvent, curTick() + pollingInterval);
}

void GpuDVFSHandler::UpdateEvent::updatePerfLevel()
{
    auto d = handler->findDomain(domainIDToSet);
    if (d) d->perfLevel(perfLevelToSet);
}

// --------------------------------------------------------------------------
// STATS DUMP
// --------------------------------------------------------------------------
int GpuDVFSHandler::dumpImportantStatsToConsole()
{
    static double prevInstTotal[64] = {0};
    static double prevNumCycles[64] = {0};
    static double edpTotal = 0.0;
    static double ed2pTotal = 0.0;
    static Tick initTime = 0;

    for (auto *cu : gpuShader->cuList) {
        int cu_idx = cuIdMap[cu];
        if (cu->stats.totalCycles.total() == prevNumCycles[cu_idx]) continue;

        double instr = cu->stats.numInstrExecuted.total();
        double deltaInstr = instr - prevInstTotal[cu_idx];
        if(deltaInstr < 0) deltaInstr = 0;

        double numCycles = cu->stats.totalCycles.total();
        double deltaNumCycles = numCycles - prevNumCycles[cu_idx];
        if(deltaNumCycles <= 0) deltaNumCycles = 1;
        
        double deltaIPC = deltaInstr / deltaNumCycles;
        double currentActivityScore = deltaIPC / TARGET_WAVE_IPC; // ACTUAL Score

        double v = cu->voltage();
        double fHz = cu->frequency();
        double fMHz = fHz / 1e6;

        double S = currentCuSensitivity[cu]; // PREDICTED Score

        double perf = predictPerf(S, fMHz); 
        double power = computePower(fMHz, v);
        double edp = computeED2P(perf, power) * perf; 
        double ed2p = computeED2P(perf, power);

        edpTotal += edp;
        ed2pTotal += ed2p;

        if (initTime == 0) initTime = curTick();

        inform("GPU_DVFS_STATS: CU: %d, clock: %lld, IPC: %.4f, ActScore: %.4f, TgtIPC: %.4f, Freq: %.0f MHz, Voltage: %.2f, EDP: %.2f, ED2P: %.2f, EDP_Total: %.2f, ED2P_Total: %.2f, Sens: %.4f"
           , cu_idx
           , curTick() - initTime
           , deltaIPC
           , currentActivityScore
           , TARGET_WAVE_IPC
           , fMHz
           , v
           , edp
           , ed2p
           , edpTotal
           , ed2pTotal
           , S 
        );

        prevInstTotal[cu_idx] = instr;
        prevNumCycles[cu_idx] = numCycles;
    }
    return 0;
}




} // namespace gem5
