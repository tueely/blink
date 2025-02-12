# blink

Blink is a project designed to help people aim better when they need to throw or place objects. It’s useful in many situations, but two of the most important areas where it can make a big difference are in medicine and for people with accessibility challenges.

Let’s start with medicine. In places like hospitals or labs, doctors and nurses often handle things that need to be thrown away carefully—like sharp tools or even radioactive waste. These items have to go into special containers to stay safe. But sometimes, these containers are far away, or getting too close might not be safe. For example, during surgery, no one wants to waste time walking across the room to toss something into a bin. Blink can help by making sure people can aim perfectly from a distance. This saves time, keeps everyone safer, and makes the whole process smoother.

Now, let’s talk about how Blink can help people with accessibility challenges. Take someone with Cerebral Palsy (CP), for instance. People with CP often have trouble with fine motor skills, which means aiming or throwing something accurately can be really hard. Imagine trying to toss a piece of trash into a bin that’s just a few steps away—but because of mobility issues, those few steps feel like miles. Walking over might not always be an option, and even if it is, it could take a lot of effort. Blink can step in here, too. It helps people aim better so they can complete everyday tasks without needing to move around as much.

Blink is designed through a method that can easily be implemented in hardware. The idea involves having a sensor or camera at the point of throwing, this can also be modified to the point of receival and simply requires a change in the z-directional component to expect a lower to higher delta instead of the opposite. The first aspect of the design involves creating a way in which the trajectory of an object can be mapped with absolute precision. Thereafter, the second aspect is to create an interface between this (precise) mapping and the trashcan or storage unit. The third and final aspect is to identify how to quickly (faster than the time it takes for the object to land) and accurately (based on the data from the precise mapping) move the storage unit to the point of the object landing.

Let's talk about the first aspect and how Blink approaches this. Firstly, its important to note here that there are multiple ways to approach this aspect and each depend on the use case. The use case for this project is a wall and a storage unit moving normally to the throwing direction and tangentially to the wall. This allows a vertical differentiation of the wall itself into pieces. Each piece would be a little smaller than the storage unit itself to avoid the potential of edge-falls. Crucially, the other approaches at hand depend on the situation. In radioactive waste for example, a 1-Dimensional approach is very unsafe and a 2-Dimensional or even 3-Dimensional approach would have to be taken. This is besides the point of this project whose goal is to simply prove that regarless of the mapping method, a translation from mapping to physical movement is possible and relevant.


Hardware Implementation of a Trashcan

TRASHCAN MOVEMENT IDEA: 
take a bucket and tape strings to both sides, have pulleys on both sides and then pull them on either side depending on where the projectile goes with reference to the trashcan itself

PLOT IDEA: 
I missed a shot at the trashcan.
Realized it was a skill issue.
Then remembered I study ECE at UofT.
So I coded a trashcan that never misses.

STACK IDEA: 
1. use python and openCV to break down a canvas into different layers and use the camera to identify which projectile is going to which layer and work on it with that
