from direct.interval.IntervalGlobal import *
from direct.task.Task import Task
from direct.showbase.ShowBase import ShowBase
from panda3d.core import load_prc_file_data

CONFIG: str = "window-type none"

async def gen_out(task):
    print(f"outputting {task.frame}")
    await Task.pause(1.)
    return task.again

if __name__ == '__main__':
    load_prc_file_data('', CONFIG)
    ShowBase()

    base.taskMgr.add(gen_out, "gen_out")

    base.run()
