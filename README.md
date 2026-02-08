# Microworld
This is a little cell survival game I'm working on!:)


It's focussed on economy and resource management.


### Basics:
From the moment you spawn in, your energy bar is emptying to keep you alive.
It starts full, but from there, you need to supply your own energy.
Fortunately, basic energy sources are plentiful (blue MANA balls), but so are 
things that demand your energy- like moving, making specialised tools, and 
of course, growing!

- Use the arrow keys to get around. 
- Press `f` to consume 1 unit of energy to make a FOOD ball. 
- Press `r` to consume 1 unit of energy to make an FRNA ball.
- Press `Spacebar` to consume the ball at the bottom of your stack.


The different ball types have different effects:
- FOOD: grow your cell physically larger- though this comes with added energy costs!
- HEAL: gain +1 health instantly
- MANA: gain +1 energy instantly
- FRNA: make proteins, growing more powerful (currently, speed up, but will be progression- and upgrade-related)
- SALT: reduce rate of energy loss over time


### TODO LIST:
- implement collisions between blobs and other blobs / surfaces
- add some surfaces / inert terrain
- add upgrades:
    - flagellum (significantly improves speed, looks cool)
    - actin tubules (unlocks horizontal gene transfer, improved movement, tech tree)
    - digestive enzymes (unlocks ability to consume smaller cells)
- add menu
- add tutorial/controls
- add enemy autonomy / multiplayer...
- nicen up the appearance with lots of nice shaders:
    - water/liquid shine and caustic effects
    - clouds/fog
    - sparkles, splatters, pops
    - HUD overlay
