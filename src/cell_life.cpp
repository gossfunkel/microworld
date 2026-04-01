#include <iostream>
//#include <AsyncTask.h>
#include "asyncTaskManager.h"
//#include <cstdint>

enum ResourceTypes {
    WATER,
    SALTS,
    SUGAR,
    CARBS,
    OILS,
    AMINO,
};

enum Ability {
    ABSORB,
    WRIGGLE,
    SWIM,
    ACIDSPRAY,
    STAB,
    HIBERNATE,
};

typedef struct Resource {
    int type;
    float qty;
    float max;    
} Resource;

typedef struct Process {
    double prepause_time;
    double time_paused;
    float time;
    bool paused;
    unsigned int input;
    unsigned int output;
    float cost;
    float yield;
} Process;

extern "C" {
    void* Cell_new(int idx, float size, int bits);
    void Cell_delete(void* cellptr);
    float Cell_get_resource(void* cellptr, unsigned int res_idx);
    int Cell_spend_resource(void* cellptr, unsigned int res_idx, float qty);
    float Cell_add_resource(void* cellptr, unsigned int res_idx, float qty);
    unsigned int Cell_add_process(void* cellptr, unsigned int in, unsigned int out, float cost, float yield, float time, int start_paused);
    //void* Process_get_task(void* proc);
}

// put these on the heap: each acts as an arena for all contained data
class Cell {
private:
    int m_idx;
    float m_size;
    Resource m_wtr;
    Resource m_slt;
    Resource m_sgr;
    Resource m_crb;
    Resource m_oil;
    Resource m_amo;
    std::vector<Process> m_metabolism;
    std::vector<PT(AsyncTask)> m_tasks;
    int m_abilities;
public:
    Cell(int idx, float size, int abilities) 
        : m_idx {idx}, m_size {size}, m_abilities {abilities} {
        std::cout << "--c> Constructing new cell!\n";
        m_wtr = Resource(ResourceTypes(WATER), 5.f, 10.f);
        m_slt = Resource(ResourceTypes(SALTS), 2.f, 10.f);
        m_sgr = Resource(ResourceTypes(SUGAR), 0.f, 10.f);
        m_crb = Resource(ResourceTypes(CARBS), 0.f, 10.f);
        m_oil = Resource(ResourceTypes(OILS),  0.f, 10.f);
        m_amo = Resource(ResourceTypes(AMINO), 0.f, 10.f);
        m_metabolism = {};
        std::cout << "--c> new cell resources: water " << m_wtr.qty << ", salts " << m_slt.qty 
                                       << ", sugar " << m_sgr.qty << ", carbs " << m_crb.qty 
                                     << ", oils " << m_oil.qty << ", amino " << m_amo.qty << ".\n";
    }

    // TODO constructor for passing in a sequence of values to set initial resources

    // RESOURCE METHODS

    Resource* get_res_ptr(unsigned int res_idx) {
        Resource* resource_ptr = nullptr;
        switch (res_idx) {
            case ResourceTypes(WATER): resource_ptr = &m_wtr; break;
            case ResourceTypes(SALTS): resource_ptr = &m_slt; break;
            case ResourceTypes(CARBS): resource_ptr = &m_crb; break;
            case ResourceTypes(SUGAR): resource_ptr = &m_sgr; break;
            case ResourceTypes(OILS):  resource_ptr = &m_oil; break;
            case ResourceTypes(AMINO): resource_ptr = &m_amo; break;
        }
        return resource_ptr;
    }

    float get_resource(unsigned int res_idx) {
        return get_res_ptr(res_idx)->qty;
    }

    // return 1 if insufficient funds
    bool spend_resource(unsigned int res_idx, float qty) {
        Resource* resource_ptr = get_res_ptr(res_idx);
        if (resource_ptr->qty < qty) return 1;
        
        resource_ptr->qty -= qty;
        return 0;
    }

    float add_resource(unsigned int res_idx, float qty) {
        Resource* resource_ptr = get_res_ptr(res_idx);
        resource_ptr->qty = std::min(resource_ptr->qty + qty, resource_ptr->max);
        return resource_ptr->qty;
    }

    // PROCESS METHODS

    // returns success/failure
    bool pause(unsigned int task_id) {
        // note time at which task is paused
        m_metabolism[task_id].prepause_time = m_tasks[task_id]->get_elapsed_time();
        if (m_metabolism[task_id].paused) return 0;
        else m_metabolism[task_id].paused = true;
        if (!m_metabolism[task_id].paused) return 1;
        return 0;
    }

    // returns success/failure
    bool resume(unsigned int task_id) {
        // keep track of how long spent paused for timekeeping
        m_metabolism[task_id].time_paused += m_tasks[task_id]->get_elapsed_time() - m_metabolism[task_id].prepause_time;
        if (!m_metabolism[task_id].paused) return 0;
        else m_metabolism[task_id].paused = false;
        if (m_metabolism[task_id].paused) return 1;
        return 0;
    }

    // returns pause state
    bool toggle_pause_task(unsigned int task_id) {
        if (m_metabolism[task_id].paused) resume(task_id);
        else pause(task_id);
        return m_metabolism[task_id].paused;
    }

    // returns success/failure
    bool do_exchange(unsigned int task_id) {
        // TODO acquire the GIL
        std::cout << "--c> Doing exchange for task " << task_id << "; Resources: "
            << m_metabolism[task_id].input << " in, " << m_metabolism[task_id].output << " out.\n";

        // fail if insufficient input resource
        if (get_resource(m_metabolism[task_id].input) < m_metabolism[task_id].cost) return 1;
        // charge input and yield to output
        spend_resource(m_metabolism[task_id].input, m_metabolism[task_id].cost);
        add_resource(m_metabolism[task_id].output, m_metabolism[task_id].yield);
        // reset pause timer
        m_metabolism[task_id].prepause_time = 0.;
        m_metabolism[task_id].time_paused = 0.;

        // TODO release the GIL

        // return success
        return 0;
    }

    unsigned int add_process(unsigned int in_type, unsigned int out_type, float cost, float yield, float time, bool start_paused) {
        // extend the metabolism vector and initialise a new process in the new field
        std::cout << "--c> Cell initialising the construction of a new process:\n";
        
        // store process data
        m_metabolism.emplace_back(Process{0., 0., time, start_paused, in_type, out_type, cost, yield});
        std::cout << "--c> Data struct made in cell's 'metabolism' vector!\n";
        // create task
        unsigned int proc_idx = m_metabolism.size()-1;
        std::cout << "--c> Creating task for metabolic process:\n";
        PT(AsyncTask) new_task = AsyncTaskManager::get_global_ptr()->add(
            [&, proc_idx](AsyncTask* task) { 
                //std::cout << "--c> task elapsed time: " << task->get_elapsed_time() 
                //            << ", and timer length: " << this->time << ".\n";
                if (task->get_elapsed_time() - this->m_metabolism[proc_idx].time_paused > this->m_metabolism[proc_idx].time) {
                    if (!this->do_exchange(proc_idx)) std::cerr << "--c> EXCHANGE FAILED!\n";
                    else {
                        return AsyncTask::DS_again;
                    }
                }
                return AsyncTask::DS_cont; // TODO throw error
                //return AsyncTask::DS_done; 
            }, "proc_task", 0);
        std::cout << "--c> Process initialised.\n";
        return proc_idx;
    }
};

// C-type binding funcs

// Cell object constructor and destructor
void* Cell_new(int idx, float size, int bits) {
    // todo use C++ casting
    return (void*) new Cell(idx, size, bits);
}
void Cell_delete(void* cellptr) {
    delete (Cell*)cellptr;
}

// resource methods
float Cell_get_resource(void* cellptr, unsigned int res_idx) {
    return ((Cell*)cellptr)->get_resource(res_idx);
}
int Cell_spend_resource(void* cellptr, unsigned int res_idx, float qty) {
    return (int)(((Cell*)cellptr)->spend_resource(res_idx, qty));
}
float Cell_add_resource(void* cellptr, unsigned int res_idx, float qty) {
    return ((Cell*)cellptr)->add_resource(res_idx, qty);
}

// process (n.b. make sure to add task to taskmgr)
unsigned int Cell_add_process(void* cellptr, unsigned int in, unsigned int out, float cost, 
                    float yield, float time, int start_paused) {
    return (((Cell*)cellptr)->add_process(in, out, cost, yield, time, (bool)start_paused));
}

/*void* Process_get_task(void* proc) {
    return (void*)(&(((Process*)proc)->update_task));
}*/
