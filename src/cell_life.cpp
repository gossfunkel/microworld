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

extern "C" {
    void* Cell_new(int idx, float size, int bits);
    void Cell_delete(void* cellptr);
    float Cell_get_water(void* cellptr);
    int Cell_spend_water(void* cellptr, float qty);
    float Cell_add_water(void* cellptr, float qty);
    float Cell_get_salts(void* cellptr);
    int Cell_spend_salts(void* cellptr, float qty);
    float Cell_add_salts(void* cellptr, float qty);
    float Cell_get_oils(void* cellptr);
    int Cell_spend_oils(void* cellptr, float qty);
    float Cell_add_oils(void* cellptr, float qty);
    float Cell_get_sugar(void* cellptr);
    int Cell_spend_sugar(void* cellptr, float qty);
    float Cell_add_sugar(void* cellptr, float qty);
    float Cell_get_carbs(void* cellptr);
    int Cell_spend_carbs(void* cellptr, float qty);
    float Cell_add_carbs(void* cellptr, float qty);
    float Cell_get_amino(void* cellptr);
    int Cell_spend_amino(void* cellptr, float qty);
    float Cell_add_amino(void* cellptr, float qty);
    void* Cell_add_process(void* cellptr, void* task_mgr_ptr, int in, int out, float cost, float yield, float time, int start_paused);
    void* Process_get_task(void* proc);
}

//AsyncTask::DoneStatus process_task(GenericAsyncTask* task, void* data);

// body
class Process {
private:
    double m_prepause_time;
    double m_time_paused;
    bool m_paused;
public:
    PT(AsyncTask) update_task;
    Resource* input;
    Resource* output;
    float cost;
    float yield;
    float time;

    // TODO FIXME type issues
    Process(PT(AsyncTaskManager) task_mgr_ptr, Resource* in, Resource* out, float cst, 
            float yld, float tm, bool start_paused)
        : update_task(task_mgr_ptr->add([this](AsyncTask* task) mutable { 
            if (task->get_elapsed_time() - this->m_time_paused < this->time)
                return AsyncTask::DS_cont;
            if (!this->do_exchange()) return AsyncTask::DS_again; // TODO throw error
            return AsyncTask::DS_done; }, "proc_task")),
          m_prepause_time(0.), m_time_paused(0.), m_paused{start_paused}, 
          input{in}, output{out}, cost{cst}, yield{yld}, time{tm} {
    }

    // returns success/failure
    bool do_exchange() {
        // TODO acquire the GIL

        // fail if insufficient resource in
        if (input->qty < cost) return 1;
        // charge input and yield to output
        input->qty -= cost;
        output->qty += yield;
        // reset pause timer
        m_prepause_time = 0.;
        m_time_paused = 0.;

        // TODO release the GIL

        // return success
        return 0;
    }

    // returns success/failure
    bool pause() {
        // note time at which task is paused
        m_prepause_time = update_task->get_elapsed_time();
        if (m_paused) return 0;
        else m_paused = true;
        if (!m_paused) return 1;
        return 0;
    }

    // returns success/failure
    bool resume() {
        // keep track of how long spent paused for timekeeping
        m_time_paused += update_task->get_elapsed_time() - m_prepause_time;
        if (!m_paused) return 0;
        else m_paused = false;
        if (m_paused) return 1;
        return 0;
    }

    // returns pause state
    bool toggle_pause() {
        m_paused = !m_paused;
        return m_paused;
    }
};

/*AsyncTask::DoneStatus process_task(GenericAsyncTask* task, void* data) {
    Process* proc = (Process*)data;
    if (task->get_elapsed_time() - proc->m_time_paused < time) return AsyncTask::DS_cont;
    bool result_exchange = proc->do_exchange();
    if (!result_exchange) return AsyncTask::DS_again;
    return AsyncTask::DS_done;
}*/

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
    int m_abilities;
public:
    Cell(int idx, float size, int abilities) 
        : m_idx {idx}, m_size {size}, m_abilities {abilities} {
        m_wtr = Resource(ResourceTypes(WATER), 5.f, 10.f);
        m_slt = Resource(ResourceTypes(SALTS), 2.f, 10.f);
        m_sgr = Resource(ResourceTypes(SUGAR), 0.f, 10.f);
        m_crb = Resource(ResourceTypes(CARBS), 0.f, 10.f);
        m_oil = Resource(ResourceTypes(OILS),  0.f, 10.f);
        m_amo = Resource(ResourceTypes(AMINO), 0.f, 10.f);
        m_metabolism = {};
    }

    // TODO constructor for passing in sequence of values for initial resources
    // TODO constructor for initialising a metabolism

    float get_water() {
        return m_wtr.qty;
    }

    // return 1 if insufficient funds
    bool spend_water(float qty) {
        if (m_wtr.qty < qty) return 1;
        
        m_wtr.qty -= qty;
        return 0;
    }

    float add_water(float qty) {
        m_wtr.qty = std::min(m_wtr.qty + qty, m_wtr.max);
        return m_wtr.qty;
    }

    float get_salts() {
        return m_slt.qty;
    }

    // return 1 if insufficient funds
    bool spend_salts(float qty) {
        if (m_slt.qty < qty) return 1;
        
        m_slt.qty -= qty;
        return 0;
    }

    float add_salts(float qty) {
        m_slt.qty = std::min(m_slt.qty + qty, m_slt.max);
        return m_slt.qty;
    }

    float get_sugar() {
        return m_sgr.qty;
    }

    // return 1 if insufficient funds
    bool spend_sugar(float qty) {
        if (m_sgr.qty < qty) return 1;
        
        m_sgr.qty -= qty;
        return 0;
    }

    float add_sugar(float qty) {
        m_sgr.qty = std::min(m_sgr.qty + qty, m_sgr.max);
        return m_sgr.qty;
    }

    float get_carbs() {
        return m_crb.qty;
    }

    // return 1 if insufficient funds
    bool spend_carbs(float qty) {
        if (m_crb.qty < qty) return 1;
        
        m_crb.qty -= qty;
        return 0;
    }

    float add_carbs(float qty) {
        m_crb.qty = std::min(m_crb.qty + qty, m_crb.max);
        return m_crb.qty;
    }

    float get_oils() {
        return m_oil.qty;
    }

    // return 1 if insufficient funds
    bool spend_oils(float qty) {
        if (m_oil.qty < qty) return 1;
        
        m_oil.qty -= qty;
        return 0;
    }

    float add_oils(float qty) {
        m_oil.qty = std::min(m_oil.qty + qty, m_oil.max);
        return m_oil.qty;
    }

    float get_amino() {
        return m_amo.qty;
    }

    // return 1 if insufficient funds
    bool spend_amino(float qty) {
        if (m_amo.qty < qty) return 1;
        
        m_amo.qty -= qty;
        return 0;
    }

    float add_amino(float qty) {
        m_amo.qty = std::min(m_amo.qty + qty, m_amo.max);
        return m_amo.qty;
    }

    Process* add_process(PT(AsyncTaskManager) task_mgr_ptr, int in_type, int out_type, float cost, float yield, float time, bool start_paused) {
        // extend the metabolism vector and initialise a new process in the new field
        Resource* in_res;
        Resource* out_res;
        switch (in_type) {
            case ResourceTypes(WATER): in_res = &m_wtr; break;
            case ResourceTypes(SALTS): in_res = &m_slt; break;
            case ResourceTypes(CARBS): in_res = &m_crb; break;
            case ResourceTypes(SUGAR): in_res = &m_sgr; break;
            case ResourceTypes(OILS):  in_res = &m_oil; break;
            case ResourceTypes(AMINO): in_res = &m_amo; break;
        }
        switch (out_type) {
            case ResourceTypes(WATER): out_res = &m_wtr; break;
            case ResourceTypes(SALTS): out_res = &m_slt; break;
            case ResourceTypes(CARBS): out_res = &m_crb; break;
            case ResourceTypes(SUGAR): out_res = &m_sgr; break;
            case ResourceTypes(OILS):  out_res = &m_oil; break;
            case ResourceTypes(AMINO): out_res = &m_amo; break;
        }
        m_metabolism.emplace_back(Process(task_mgr_ptr, in_res, out_res, cost, yield, time, start_paused));
        return &m_metabolism.at(m_metabolism.size()-1);
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

// water 
float Cell_get_water(void* cellptr) {
    return ((Cell*)cellptr)->get_water();
}
int Cell_spend_water(void* cellptr, float qty) {
    return (int)(((Cell*)cellptr)->spend_water(qty));
}
float Cell_add_water(void* cellptr, float qty) {
    return ((Cell*)cellptr)->add_water(qty);
}

// salts
float Cell_get_salts(void* cellptr) {
    return ((Cell*)cellptr)->get_salts();
}
int Cell_spend_salts(void* cellptr, float qty) {
    return (int)(((Cell*)cellptr)->spend_salts(qty));
}
float Cell_add_salts(void* cellptr, float qty) {
    return ((Cell*)cellptr)->add_salts(qty);
}

// oils
float Cell_get_oils(void* cellptr) {
    return ((Cell*)cellptr)->get_oils();
};
int Cell_spend_oils(void* cellptr, float qty) {
    return (int)(((Cell*)cellptr)->spend_oils(qty));
};
float Cell_add_oils(void* cellptr, float qty) {
    return ((Cell*)cellptr)->add_oils(qty);
};

// sugar
float Cell_get_sugar(void* cellptr) {
    return ((Cell*)cellptr)->get_sugar();
}
int Cell_spend_sugar(void* cellptr, float qty) {
    return (int)(((Cell*)cellptr)->spend_sugar(qty));
}
float Cell_add_sugar(void* cellptr, float qty) {
    return ((Cell*)cellptr)->add_sugar(qty);
}

// carbs
float Cell_get_carbs(void* cellptr) {
    return ((Cell*)cellptr)->get_carbs();
}
int Cell_spend_carbs(void* cellptr, float qty) {
    return (int)(((Cell*)cellptr)->spend_carbs(qty));
}
float Cell_add_carbs(void* cellptr, float qty) {
    return ((Cell*)cellptr)->add_carbs(qty);
}

// amino
float Cell_get_amino(void* cellptr) {
    return ((Cell*)cellptr)->get_amino();
}
int Cell_spend_amino(void* cellptr, float qty) {
    return (int)(((Cell*)cellptr)->spend_amino(qty));
}
float Cell_add_amino(void* cellptr, float qty) {
    return ((Cell*)cellptr)->add_amino(qty);
}

// process (n.b. make sure to add task to taskmgr)
void* Cell_add_process(void* cellptr, void* task_mgr_ptr, int in, int out, float cost, 
                    float yield, float time, int start_paused) {
    return (void*)(((Cell*)cellptr)->add_process((AsyncTaskManager*)task_mgr_ptr, in, 
                    out, cost, yield, time, (bool)start_paused));
}

void* Process_get_task(void* proc) {
    return (void*)(&(((Process*)proc)->update_task));
}
