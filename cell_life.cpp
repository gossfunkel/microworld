#include <cstdint>
#include <genericAsyncTask.h>
#include "asyncTaskManager.h"

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

class Process {
private:
    double m_prepause_time;
    double m_time_paused;
    bool m_paused;
public:
    PT(GenericAsyncTask) update_task;
    Resource* input;
    Resource* output;
    float cost;
    float yield;
    float time;

    Process(Resource* in, Resource* out, float cst, 
            float yld, float tm, bool start_paused)
        : update_task(&process_task), m_prepause_time(0.), m_time_paused(0.), m_paused{start_paused}, 
          update_task{task}, input{in}, output{out}, cost{cst}, yield{yld}, time{tm}, {
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

    AsyncTask::DoneStatus process_task(GenericAsyncTask *task, void *data) {
        if (task->get_elapsed_time() - m_time_paused < time)
            return AsyncTask::DS_cont;
        
        if (!do_exchange()) return AsyncTask::DS_again;
        return AsyncTask::DS_done;
    }
};

// put these on the heap: each acts as an arena for all contained data
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
        m_wtr.qty = std::max(m_wtr.qty + qty, m_wtr.max);
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
        m_slt.qty = std::max(m_slt.qty + qty, m_slt.max);
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
        m_sgr.qty = std::max(m_sgr.qty + qty, m_sgr.max);
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
        m_crb.qty = std::max(m_crb.qty + qty, m_crb.max);
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
        m_oil.qty = std::max(m_oil.qty + qty, m_oil.max);
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
        m_amo.qty = std::max(m_amo.qty + qty, m_amo.max);
        return m_amo.qty;
    }

    void add_process(int in_type, int out_type, float cost, float yield, float time) {
        // extend the metabolism vector and initialise a new process in the new field
        m_metabolism.emplace_back(Process(in_type, out_type, cost, yield, time, false));
    }
}