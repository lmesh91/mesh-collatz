# mesh-collatz
Code for my research project on a novel generalization of the Collatz sequence: Mesh-Collatz sequences.

If we take the Collatz sequence:
$$C(x)=3x+1\text{ if }x\equiv 0\text{ mod }2,\frac{x}{2}\text{ if }x\equiv 1\text{ mod }2$$

The Mesh-Collatz sequences modify this slightly:
$$C(x,m)=3x+1\text{ if }x\equiv 0\text{ mod }2,\frac{x}{2}+m\text{ if }x\equiv 1\text{ mod }2$$

This repository contains code that computes the average delay of Mesh-Collatz sequences, and records other interesting statistics.

CUDA is required to run this program. I haven't tested it on other platforms, and was tested only on an  RTX 3080, so don't be surprised if it crashes unless you have a powerful GPU.
