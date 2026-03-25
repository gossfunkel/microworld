#include <bitset>
#include <cstdint>

enum ResourceTypes {
    WATER,
    SALTS,
    SUGAR,
    CARBS,
    OILS,
    AMINO,
};

namespace Abilities {
    enum Ability {
        ABSORB,
        WRIGGLE,
        SWIM,
        ACIDSPRAY,
        STAB,
        HIBERNATE,
    };
}

typedef struct Resource {
    uint_fast32_t type;
    float qty;
    float max;    
} Resource;

typedef struct Process {
    PT(GenericAsyncTask) m_update_task;
    Resource* input;
    Resource* output;
    float cost;
    float yield;
    float time;
    bool paused;
} Process;

// put these on the heap: acts as an arena for all contained data
class Cell {
private:
    uint_fast64_t m_idx;
    float m_size;
    Resource m_wtr;
    Resource m_slt;
    Resource m_sgr;
    Resource m_crb;
    Resource m_oil;
    Resource m_amo;
    std::vector<Process> m_metabolism;
    std::bitset<NUM_ABILITIES> m_abilities;
public:
    Cell(uint_fast64_t idx, float size, std::bitset<NUM_ABILITIES> abilities) 
        : m_idx {idx}, m_size {size}, m_abilities {abilities} {
        m_wtr = (Resource){ResourceTypes(WATER), 5.f, 10.f};
        m_slt = (Resource){ResourceTypes(SALTS), 2.f, 10.f};
        m_sgr = (Resource){ResourceTypes(SUGAR), 0.f, 10.f};
        m_crb = (Resource){ResourceTypes(CARBS), 0.f, 10.f};
        m_oil = (Resource){ResourceTypes(OILS),  0.f, 10.f};
        m_amo = (Resource){ResourceTypes(AMINO), 0.f, 10.f};
        m_metabolism = {};
    }

    // TODO constructor for passing in sequence of values for initial resources
    // TODO constructor for initialising a metabolism

    bool add_process(int in_type, int out_type, float cost, float yield, float time) {
        int initial_processes = m_metabolism.size();
        // extend the metabolism vector and initialise a new process in the new field
        m_metabolism.emplace_back(
                Process{&process_task, in_type, out_type, cost, yield, time, false}
            );
        // return error value if list has not extended
        if (m_metabolism.size <= initial_processes) return 1;
        // else return success
        return 0;
    }
};

// returns success/failure
bool do_exchange(Process* proc) {
    // fail if insufficient resource in
    if (proc->input->qty < proc->cost) return 1;
    // charge input and yield to output
    proc->input->qty -= proc->cost;
    proc->output->qty += proc->yield;
    // reset timer

    // return success
    return 0;
}

// returns success/failure
bool pause_process(Process* proc) {
    if (proc->paused) return 0;
    else proc->paused = true;
    if (!proc->paused) return 1;
    return 0;
}

// returns success/failure
bool run_process(Process* proc) {
    if (!proc->paused) return 0;
    else proc->paused = false;
    if (proc->paused) return 1;
    return 0;
}

// returns pause state
bool toggle_process(Process* proc) {
    proc->paused = !proc->paused;
    return proc->paused;
}

AsyncTask::DoneStatus process_task(GenericAsyncTask *task, void *data) {
    if (task->get_elapsed_time() < (Process*)data->time)
        return AsyncTask::DS_cont;
    
    do_exchange((Process*)data);
    return AsyncTask::DS_done;
}