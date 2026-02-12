from panda3d.core import (
    TextNode, CardMaker, ColorBlendAttrib, TransparencyAttrib,
)
#from mol import MOLTYPE


def setup_ui():
    # player mol counter
    base.player_mols_count = TextNode("p1_mol_count")
    base.player_mols_count.setTextColor(0,0,0,1)
    base.player_mols_count.setTextScale(0.06)
    base.player_mols_count.setText("0")
    player_mols_count_np = render2d.attach_new_node(base.player_mols_count)
    player_mols_count_np.set_pos((.82,0.,.85))

    # TODO stat bar(s) for water/salinity
    base.player_hydration = TextNode("p1_hydration")
    base.player_hydration.setTextColor(.4,.4,1,1)
    base.player_hydration.setTextScale(0.06)
    base.player_hydration.setText("0")
    player_hydration_np = render2d.attach_new_node(base.player_hydration)
    player_hydration_np.set_pos((.82,0.,.7))

    base.player_salinity = TextNode("p1_salinity")
    base.player_salinity.setTextColor(1,1,1,1)
    base.player_salinity.setTextScale(0.06)
    base.player_salinity.setText("0")
    player_salinity_np = render2d.attach_new_node(base.player_salinity)
    player_salinity_np.set_pos((.82,0.,.55))

    # player resource counters
    base.player_carb = TextNode("p1_carb_count")
    base.player_carb.setTextColor(1,1,0,1)
    base.player_carb.setTextScale(0.06)
    base.player_carb.setText("0")
    player_carb_np = render2d.attach_new_node(base.player_carb)
    player_carb_np.set_pos((.68,0.,-.872))

    base.player_oil = TextNode("p1_oil_count")
    base.player_oil.setTextColor(0,1,0,1)
    base.player_oil.setTextScale(0.06)
    base.player_oil.setText("0")
    player_oil_np = render2d.attach_new_node(base.player_oil)
    player_oil_np.set_pos((.76,0.,-.872))

    base.player_amino = TextNode("p1_amino_count")
    base.player_amino.setTextColor(1,0,1,1)
    base.player_amino.setTextScale(0.06)
    base.player_amino.setText("0")
    player_amino_np = render2d.attach_new_node(base.player_amino)
    player_amino_np.set_pos((.82,0.,-.872))

    bar_maker = CardMaker("bars")
    bar_maker.set_frame(0.,1.,0.,.1,)

    #    render   = incoming_col * A - framebuffer_col * B
    bar_text_cba  = ColorBlendAttrib.make(ColorBlendAttrib.M_subtract, 
                                          ColorBlendAttrib.O_one, 
                                          ColorBlendAttrib.O_one)
    bar_text_trat = TransparencyAttrib.make(TransparencyAttrib.M_binary)

    # energy HUD
    base.nrg_bar = render2d.attach_new_node(bar_maker.generate())
    base.nrg_bar.set_pos((-.98,0.,-.9))
    base.nrg_bar.set_texture(loader.loadTexture("teal.png"))

    base.nrg_label = TextNode("nrg_label")
    base.nrg_label.set_attrib(bar_text_trat, 1000)
    base.nrg_label.setTextColor(.5,1.,1.,.8)
    base.nrg_label.setTextScale(0.07)
    base.nrg_label.setText("ENERGY:")
    base.nrg_label.setAttrib(bar_text_cba)
    nrg_label_np = render2d.attach_new_node(base.nrg_label)
    nrg_label_np.set_pos((-.97,0.,-.872))

    # energy HUD value output
    base.nrg_label_v = TextNode("0")
    base.nrg_label_v.set_attrib(bar_text_trat, 1000)
    base.nrg_label_v.setTextColor(.6,1.,1.,.8)
    base.nrg_label_v.setTextScale(0.072)
    base.nrg_label_v.setAttrib(bar_text_cba)
    nrg_label_v_np = render2d.attach_new_node(base.nrg_label_v)
    nrg_label_v_np.set_pos((-.68,0.,-.872))

    # health HUD
    base.hp_bar = render2d.attach_new_node(bar_maker.generate())
    base.hp_bar.set_pos((-.98,0.,-.75))
    base.hp_bar.set_texture(loader.loadTexture("green.png"))

    base.hp_label = TextNode("hp_label")
    base.hp_label.set_attrib(bar_text_trat, 1000)
    base.hp_label.setTextColor(.5,1.,.5,.8)
    base.hp_label.setTextScale(0.07)
    base.hp_label.setText("HEALTH:")
    base.hp_label.setAttrib(bar_text_cba)
    hp_label_np = render2d.attach_new_node(base.hp_label)
    hp_label_np.set_pos((-.97,0.,-.722))

    # health HUD value output
    base.hp_label_v = TextNode("0")
    base.hp_label_v.set_attrib(bar_text_trat, 1000)
    base.hp_label_v.setTextColor(.6,1.,.6,.8)
    base.hp_label_v.setTextScale(0.072)
    base.hp_label_v.setAttrib(bar_text_cba)
    hp_label_v_np = render2d.attach_new_node(base.hp_label_v)
    hp_label_v_np.set_pos((-.68,0.,-.722))

def update(task):
    base.cam.setPos(base.p1.pos() + base.CAM_POS)                   # camera follows p1

    base.player_mols_count.setText(str(len(base.p1.mols)))          # update UI
    base.player_hydration.setText(str(base.p1.hydration))
    base.player_salinity.setText(str(base.p1.salinity))
    base.player_carb.setText(str(base.p1.carbs))
    base.player_oil.setText(str(base.p1.oils))
    base.player_amino.setText(str(base.p1.aminos))

    base.nrg_label_v.setText(str(base.p1.nrg)[:4]+"/"+str(base.p1.max_hp)[:2])
    base.hp_label_v.setText(str(base.p1.hp)[:4]+"/"+str(base.p1.max_nrg)[:2])
    base.nrg_bar.set_scale((base.p1.nrg/base.p1.max_nrg)*.75,1.,1.)
    base.hp_bar.set_scale((base.p1.hp/base.p1.max_hp)*.75,1.,1.)
    return task.cont