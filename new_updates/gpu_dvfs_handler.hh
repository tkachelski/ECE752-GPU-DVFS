#ifndef __SIM_GPU_DVFS_HANDLER_HH__
#define __SIM_GPU_DVFS_HANDLER_HH__

#include "params/GpuDVFSHandler.hh"
#include "sim/sim_object.hh"
#include "sim/clock_domain.hh"
#include "sim/eventq.hh"
#include <map>

// GPU Headers required to spy on Wavefronts
#include "gpu-compute/shader.hh"
#include "gpu-compute/compute_unit.hh"
#include "gpu-compute/wavefront.hh"

namespace gem5
{

/**
 * GpuDVFSHandler
 * A specialized handler for managing GPU Dynamic Voltage and Frequency Scaling (DVFS).
 * * IMPLEMENTATION STRATEGY: PCStall (Predict, Don't React)
 * --------------------------------------------------------
 * Instead of reacting to past utilization history (which is often too late), 
 * this handler scans the instantaneous state (Program Counter distribution) 
 * of all active wavefronts. 
 * * - High PC Concentration implies synchronization/stalls (Barrier/Memory).
 * - Low PC Concentration implies independent progress (ALU/Throughput).
 */
class GpuDVFSHandler : public SimObject
{
  public:
    typedef GpuDVFSHandlerParams Params;
    GpuDVFSHandler(const Params &p);

    // Standard gem5 typedefs
    typedef SrcClockDomain::DomainID DomainID;
    typedef SrcClockDomain::PerfLevel PerfLevel;

    /**
     * startup()
     * Called by gem5 after all objects are created but before simulation starts.
     * Use this to schedule the first iteration of our decision loop.
     */
    void startup() override;

  private:
    // Container to store pointers to the clock domains
    typedef std::map<DomainID, SrcClockDomain*> Domains;
    Domains domains;
    
    SrcClockDomain *sysClkDomain;
    bool enableHandler;
    bool printToScreen;
    Tick _transLatency;

    // Pointer to the real GPU hardware
    Shader *gpuShader;

    // Main event wrapper for the decision loop
    EventFunctionWrapper decisionEvent;
    
    // ----------------------------------------------------------------------
    // Core Logic Functions
    // ----------------------------------------------------------------------

    /**
     * runDecisionLoop()
     * The "Governor" logic. 
     * 1. Aggregates global GPU state.
     * 2. Calculates Wavefront PC Concentration.
     * 3. Predicts Stall vs. Busy.
     * 4. Actuates frequency changes.
     */
    void runDecisionLoop();

    /**
     * scanGlobalWavefrontState()
     * Scans ALL Compute Units and ALL Wavefronts.
     * Returns a Histogram: <PC Address, Count of Wavefronts at this PC>
     * This provides the "Signature" of the workload at this exact tick.
     */
    std::map<Addr, int> scanGlobalWavefrontState(); 

    /**
     * findDomain()
     * Helper to retrieve a clock domain object given its ID.
     */
    SrcClockDomain *findDomain(DomainID domain_id) const;

    /**
     * UpdateEvent
     * A specialized event that performs the actual physical clock change.
     * Separate the "Decision" (logic) from the "Update" (actuation) to 
     * allow for modeling transition latency if desired.
     */
    struct UpdateEvent : public Event
    {
        GpuDVFSHandler *handler;       
        DomainID domainIDToSet;        
        PerfLevel perfLevelToSet;      

        UpdateEvent() : Event(Default_Pri, AutoDelete), handler(nullptr) {}
        
        void process() override { updatePerfLevel(); }
        void updatePerfLevel();
        
        const char *description() const override { return "GPU DVFS Update Perf Level"; }
    };
};

} // namespace gem5

#endif // __SIM_GPU_DVFS_HANDLER_HH__


