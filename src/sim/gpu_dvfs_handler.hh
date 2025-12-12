#ifndef __SIM_CU_REACTIVE_GPU_DVFS_HANDLER_HH__
#define __SIM_CU_REACTIVE_GPU_DVFS_HANDLER_HH__

#include "params/GpuDVFSHandler.hh"
#include "sim/sim_object.hh"
#include "sim/clock_domain.hh"
#include "sim/eventq.hh"
#include <map>
#include <vector>

// GPU Headers
#include "gpu-compute/shader.hh"
#include "gpu-compute/compute_unit.hh"
#include "gpu-compute/wavefront.hh"

namespace gem5
{

/**
 * GpuDVFSHandler (CU-Based Reactive Version)
 * ----------------------------------------------------------------------
 * METRIC STANDARDIZATION:
 * Uses the Sensitivity, Power, and ED2P models
 *
 * STRATEGY:
 * - Measure average Sensitivity (IPC/Target) of the PREVIOUS epoch.
 * - Assume CURRENT epoch will be identical (Reactive).
 * - Optimize for ED2P.
 */
class GpuDVFSHandler : public SimObject
{
  public:
    typedef GpuDVFSHandlerParams Params;
    GpuDVFSHandler(const Params &p);

    typedef SrcClockDomain::DomainID DomainID;
    typedef SrcClockDomain::PerfLevel PerfLevel;

    void startup() override;

  private:
    // ----------------------------------------------------------------------
    // Metrics & History
    // ----------------------------------------------------------------------
    std::map<Wavefront*, Tick> wfCreationTick;
    std::map<Wavefront*, double> lastWfInstCount;
    std::map<Wavefront*, double> lastWfSchCycles;
    
    // Per-CU Sensitivity for Stats/Decision
    std::map<ComputeUnit*, double> currentCuSensitivity; 

    // Mappings
    std::map<ComputeUnit*, DomainID> cuToDomain;
    std::map<ComputeUnit*, int> cuIdMap;

    // ----------------------------------------------------------------------
    // Standardized Model Constants
    // ----------------------------------------------------------------------
    static const int NUM_LEVELS = 3;
    // 4GHz (High), 2GHz (Balanced), 1GHz (Low)
    double freqsMHz[NUM_LEVELS] = {4000, 2000, 1000};
    double volts[NUM_LEVELS] = {1.0, 0.9, 0.8};

    const double C_DYNAMIC = 1.0; 
    const double A_ACTIVITY = 1.0; 

    // Gem5 Members
    typedef std::map<DomainID, SrcClockDomain *> Domains;
    Domains domains;
    SrcClockDomain *sysClkDomain;
    bool enableHandler;
    Tick _transLatency;
    Tick pollingInterval;
    Shader *gpuShader;
    EventFunctionWrapper decisionEvent;

    // Core Functions
    void runDecisionLoop();
    int checkIfGPUIsRunning();
    int dumpImportantStatsToConsole();
    SrcClockDomain *findDomain(DomainID domain_id) const;

    // Metrics & Updates
    double computeSensitivity(double deltaInsts, double deltaSchCycles);
    double predictPerf(double S, double fMHz, double fNomMHz = 2000.0);
    double computePower(double fMHz, double v);
    double computeED2P(double perf, double power);
    
    PerfLevel chooseBestLevel(double cuSumS, int cuID);

    struct UpdateEvent : public Event
    {
        GpuDVFSHandler *handler;
        DomainID domainIDToSet;
        PerfLevel perfLevelToSet;
        UpdateEvent() : Event(Default_Pri, AutoDelete), handler(nullptr) {}
        void process() override { updatePerfLevel(); }
        void updatePerfLevel();
        const char *description() const override { return "GPU DVFS Update"; }
    };
};

} // namespace gem5

#endif // __SIM_CU_REACTIVE_GPU_DVFS_HANDLER_HH__

