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
- Press `g` to use 1 carb and 1 oil to grow your cell.
- Press `h` to rapidly heal your cell for 1 oil.
- Use `j` and `k` to select from collected mols (floating above your cell).
- Press `spacebar` to consume the selected mol.


The different ball types have different effects:
- WATER: maintain hydration levels to prevent health loss and enable optimal energy generation
- SALT: reduce rate of energy loss over time. Caution: high salt levels are bad for your health!
- SUGAR: quick energy
- CARBS: metabolic energy source- maintain supplies so your metabolism can keep you alive
- OILS: heal or build your cell defenses- though this comes with added energy costs!
- AMINO: make proteins, growing larger and more powerful 


### TODO LIST:
- finish implementing a tessellation shader to smooth the cell outlines
- polish resources; make resources fade over time after consumption
- figure out how to implement collisions between cells and other cells / surfaces
- replace keyboard control system- so that multiple keys can be pressed at once
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

## BUILD

Rebuilding the C++ library is possible, but I've only just figured out how to 
get it to build on my own system!  Hahaha

The library is currently configured for Windows 64bit *only*!
```sh
rm -r build
mkdir build && cd build
cmake .. -A x64
cmake --build . --config RelWithDebInfo
```

## History

#### 0.0.4
Created a C++ dynamic library so I could manage my resources with pointers.  
Currently feels like the cost/benefit balance might have been off, at this point, but I have just slogged through making the build system etc.  
The key to making this an advantage is using the C++ library for efficient data-oriented design for the heart of the economy engine in the game.  

#### 0.0.3
Reworked the mol system to `WATER`, `SALT`, `SUGAR`, `CARBS`, `OILS`, and `AMINO`.
Created system by which consumed mols fill various stocks or balance certain values.
Added basic metabolic system- cell turns consumed CARBS into energy. Growth, healing, and [speed boosts] are now abilities that cost resources.
Player can now select between the mols they've collected with the `j` and `k` keys.
[The cell models are now smoother due to the use of a tessellation shader]

#### 0.0.2
Made it all into cells and 'mols'. Created the MOLTYPES.
Added ability to 'consume' collected mols with `spacebar`, with gold `FOOD` type to grow the cell radius, or purple `FRNA` type to increase cell speed.
Created a basic heads-up-display with health and mana bars.
Set up system of depleting mana that can be topped up by consuming teal `MANA` mols.
User can now zoom in and out with the scrollwheel, and click and drag to move the camera around.

#### 0.0.1
Created blobs that wobble about when their nodepath is moved.
Added some controls to allow the player to move the blob about on a 2D plane.
Added glowing balls and the ability to 'collect' them (they become affiliated with the blobs on collision).
Moved blobs into a vertex shader and learned about vertex pulling. Broke shader generator; no more easy glow or antialiasing!
