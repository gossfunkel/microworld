from panda3d.core import (
    TextNode, CardMaker, ColorBlendAttrib, TransparencyAttrib,
)
#from mol import MOLTYPE


def setup_ui():
    # player mol counter
    base.p1_label_v = TextNode("0")
    base.p1_label_v.setTextColor(1,1,1,1)
    base.p1_label_v.setTextScale(0.1)
    p1_label_v_np = aspect2d.attach_new_node(base.p1_label_v)
    p1_label_v_np.set_pos((1.3,0.,.85))

    # TODO player resource counters

    bar_maker = CardMaker("bars")
    bar_maker.set_frame(0.,1.,0.,.1,)

    # TODO stat bar(s) for water/salinity

    #    render   = incoming_col * A - framebuffer_col * B
    bar_text_cba  = ColorBlendAttrib.make(ColorBlendAttrib.M_subtract, 
                                          ColorBlendAttrib.O_one, 
                                          ColorBlendAttrib.O_one)
    bar_text_trat = TransparencyAttrib.make(TransparencyAttrib.M_binary)

    # energy HUD
    base.nrg_bar = aspect2d.attach_new_node(bar_maker.generate())
    base.nrg_bar.set_pos((-1.4,0.,-.9))
    base.nrg_bar.set_texture(loader.loadTexture("teal.png"))

    base.nrg_label = TextNode("nrg_label")
    base.nrg_label.set_attrib(bar_text_trat, 1000)
    base.nrg_label.setTextColor(.5,1.,1.,.8)
    base.nrg_label.setTextScale(0.078)
    base.nrg_label.setText("ENERGY:")
    base.nrg_label.setAttrib(bar_text_cba)
    nrg_label_np = aspect2d.attach_new_node(base.nrg_label)
    nrg_label_np.set_pos((-1.35,0.,-.872))

    # energy HUD value output
    base.nrg_label_v = TextNode("0")
    base.nrg_label_v.set_attrib(bar_text_trat, 1000)
    base.nrg_label_v.setTextColor(.6,1.,1.,.8)
    base.nrg_label_v.setTextScale(0.08)
    base.nrg_label_v.setAttrib(bar_text_cba)
    nrg_label_v_np = aspect2d.attach_new_node(base.nrg_label_v)
    nrg_label_v_np.set_pos((-.97,0.,-.87))

    # health HUD
    base.hp_bar = aspect2d.attach_new_node(bar_maker.generate())
    base.hp_bar.set_pos((-1.4,0.,-.75))
    base.hp_bar.set_texture(loader.loadTexture("green.png"))

    base.hp_label = TextNode("hp_label")
    base.hp_label.set_attrib(bar_text_trat, 1000)
    base.hp_label.setTextColor(.5,1.,.5,.8)
    base.hp_label.setTextScale(0.078)
    base.hp_label.setText("HEALTH:")
    base.hp_label.setAttrib(bar_text_cba)
    hp_label_np = aspect2d.attach_new_node(base.hp_label)
    hp_label_np.set_pos((-1.35,0.,-.722))

    # health HUD value output
    base.hp_label_v = TextNode("0")
    base.hp_label_v.set_attrib(bar_text_trat, 1000)
    base.hp_label_v.setTextColor(.6,1.,.6,.8)
    base.hp_label_v.setTextScale(0.08)
    base.hp_label_v.setAttrib(bar_text_cba)
    hp_label_v_np = aspect2d.attach_new_node(base.hp_label_v)
    hp_label_v_np.set_pos((-.97,0.,-.72))

def update(task):
    base.cam.setPos(base.p1.pos() + base.CAM_POS)                    # camera follows p1
    # print(f"Cellpos: {base.p1.pos}; cam pos: {base.cam.getPos()}")
    base.p1_label_v.setText(str(len(base.p1.mols)))             # update UI
    #base.p2_label_v.setText(str(len(base.p2.mols)))
    base.nrg_label_v.setText(str(base.p1.nrg)[:4]+"/"+str(base.p1.max_hp)[:2])
    base.hp_label_v.setText(str(base.p1.hp)[:4]+"/"+str(base.p1.max_nrg)[:2])
    base.nrg_bar.set_scale(base.p1.nrg/base.p1.max_nrg,1.,1.)
    base.hp_bar.set_scale(base.p1.hp/base.p1.max_hp,1.,1.)
    return task.cont