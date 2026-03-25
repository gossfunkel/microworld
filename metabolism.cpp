#include "asyncTaskManager.h"

typedef enum {
    WATER,
    SUGAR,
    CARB,
    OILS,
    AMINO,
    SALT
} moltype;

float get_time();

class Metabolism {
private:
    void* m_cell {};       // pointer to host cell TODO is this necessary?
    moltype m_in_type {};  // type being consumed TODO is this necessary?
    float m_in_qty {};     // how much must be consumed
    //void* m_in_from {};    // pointer to variable containing input resource
    moltype m_out_type {}; // type being produced TODO is this necessary?
    float m_out_qty {};    // how much will be produced
    //void* m_out_from {};   // pointer to variable containing output resource
    //float m_out_max {};    // maximum quantity of output resource storeable TODO cell should do this
    float m_time {};       // length of process
    float m_prev_time {};  // time at prev timestep
    float m_elapsed {0};    // time elapsed so far
    bool m_paused {1};     // is this metabolism paused?
    PT(GenericAsyncTask) m_update_task;

public:
    Metabolism(Cell* cell, moltype in_type, float in_qty, moltype out_type, float out_qty, float time) 
                : m_cell {cell}, m_in_type {in_type}, m_in_qty {in_qty},
                                 m_out_type {out_type}, m_out_qty {out_qty}, m_time {time} {
        m_prev_time = get_time();
        PT(AsyncTaskManager) task_mgr = AsyncTaskManager::get_global_ptr();
        m_update_task = new GenericAsyncTask("update_metabolism", &update, nullptr);
        task_mgr->add(m_update_task);
    }
    ~Metabolism() { // FIXME is this a destructor? lol
        m_update_task->remove();
    }


    bool is_paused() {
        return m_paused;
    }

    bool toggle_pause() {
        m_paused = ! m_paused;
        return m_paused;
    }

    void pause() {
        m_paused = 1;
    }

    void unpause() {
        m_paused = 0;
    }

    AsyncTask::DoneStatus update(GenericAsyncTask *task, void *data) {
        if (!m_paused) {
            if (m_in_qty <= m_cell->m_in_from) {
                pause();
                return t.cont;
            }
            m_elapsed += get_time() - m_prev_time;
        } else {
            unpause();
        }
    }
}