# Microworld
Microworld is an economy and resource management game about microbial life.

Collect 'mols' (resources) to balance your metabolism to grow and upgrade your cell!


### Basics:
From the moment you spawn in, your energy bar is emptying to keep you alive.
It starts full, but from there, you need to supply your own energy.
Fortunately, basic resources are plentiful, but so are 
things that demand your energy- like moving, making specialised tools, and 
of course, growing!

- Use the arrow keys to get around. 
- Press `f` to consume 1 unit of energy to make a FOOD mol. 
- Press `r` to consume 1 unit of energy to make an FRNA mol.
- Use `j` and `k` to select a gathered mol
- Press `Spacebar` to consume the selected mol


The different ball types have different effects:
- SUGAR: quick energy
- CARBS: metabolic energy source- maintain supplies so your metabolism can keep you alive
- OILS: gain health immediately, or build your cell defenses- though this comes with added energy costs!
- WATER: maintain hydration levels to prevent health loss and enable optimal energy generation
- AMINO: make proteins, growing larger and more powerful 
- SALT: reduce rate of energy loss over time. Caution: high salt levels are bad for your health!


### TODO LIST:
- finish implementing a tessellation shader to smooth the cell outlines
- polish resources; sort HUD with display for all resources, make resources fade over time after consumption, rework FRNA into AMINOS, MANA into SUGAR, FOOD into CARBS, HEAL into OILS, and add WATER. 
- figure out how to implement collisions between cells and other cells / surfaces
- add more inert terrain
- add genes (upgrades):
    - flagellum (significantly improves speed, looks cool)
    - actin tubules (unlocks horizontal gene transfer, improved mobility, tech tree pathway)
    - gene transfer (gain the ability to steal genes from other)
    - digestive enzymes (improves metabolism, and unlocks ability to consume smaller cells)
    - citric acid cycle (significantly improved metabolism)
    - sodium/calcium channels (allows salt levels to reduce when too high)
    - amylogenesis (unlocks ability to passively convert stored sugars to carbs)
    - cell wall (significantly improved defenses)
    - mitochondria (significantly improved metabolism)
    - DNA (defense against low-level genetic attacks and reduced loss of aminos)
    - toxin secretion (unlocks ability to release toxins that damage other cells)
    - ER (unlock more protein options)
    - hostile plasmids (unlocks production of genes that can be delivered to enemies)
    - antigens (defense against viral/RNA attack)
    - chromosomes (defense against higher-level genetic attacks)
    - hormone signals (unlocks ability to reduce other cell hostility)
    - heme (reduces water needs and energy loss rate)
- add menu
- add tutorial/controls
- add enemy autonomy / multiplayer...
- nicen up the appearance with lots of nice shaders:
    - water/liquid shine and caustic effects
    - some background murky bokeh
    - antialiasing
    - clouds/fog
    - sparkles, splatters, pops, glow
    - HUD overlay
- add some rare unique situational upgrades, like:
    - fungal symbiosis
    - adaptation to environment (like achievements, has requirements)
    - mutations
