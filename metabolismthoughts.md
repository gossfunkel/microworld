updated 25 march 2026:


Resources:
- are accessed randomly
- are accessed independently of each other due to async
- have a small footprint in memory
- are few in number (per cell)

The Cell object acts as an arena for all associated data; when a cell despawns, 
    it takes all the resources, processes, and references below it in the tree 
    with it. 

A linked list / pointer-reference-based approach works well here with these 
    data sizes and access patterns. 



============= original notes (~jan 2026):

the picture in my head is: 
- ideal case: user turns on the process, turns 1 resource `a` into 1 resource `b` , and then switches it off. Interval should run to completion, and then end
- prepared-for case: user turns on the process, then turns of the process before the end. The interval should terminate when the process is turned off
- prepared-for case: user turns on process, but does not have the resource required. The interval should stay suspended until the resource is present, then start.
- edge case: user turns on the process, then turns it off before it completes, then turns it back on. Interval should pause when turned off, and then continue when turned on again

class Metabolism:
    def __init__(self, cell, res_in, needed, res_out, given, time):
        self.cell         = cell                                    # owner of this metabolism
        self.resource_in  = res_in                                  # pointer to cell's resource counter - input
        self.needed       = needed                                  # quantity needed to produce
        self.resource_out = res_out                                 # pointer to cell's resource counter - output
        self.given        = given                                   # quantity of resource produced
        self.time         = time                                    # how long the process takes
        self.seq          = Sequence(                               # the Sequence doing the waiting for us
            Wait(time),
            Func(do_exchange)
        )
        base.taskMgr.add(update)
        self.seq.loop()

    def update(self, task):
        if (self.resource_in < self.needed):
            self.seq.pause()
        return task.cont

    def do_exchange(self):
        self.cell.resource_in -= self.needed
        self.cell.resource_out += self.given

    def update_metabolic_rate(self, new_time):
        self.time = new_time
        self.seq  = Sequence(                                       # make a fresh sequence with the new time
            Wait(time),
            Func(do_exchange)
        )
        self.seq.loop()

    def pause(self):
        self.seq.pause()

    def resume(self):
        self.seq.resume()



if i do it through the cell:

def resume_metabolism():
    base.p1.metabolism_seq.resume()

def pause_metabolism():
    base.p1.metabolism_seq.pause()
