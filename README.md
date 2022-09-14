# mesh-collatz
Code for my research project on the Mesh-Collatz sequences.

If we take the Collatz sequence:
$$C(x)=3x+1\text{ if }x\equiv 0\text{ mod }2,\frac{x}{2}\text{ if }x\equiv 1\text{ mod }2$$

The Mesh-Collatz sequences modify this slightly:
$$M_m(x)=3x+1\text{ if }x\equiv 0\text{ mod }2,\frac{x}{2}+m\text{ if }x\equiv 1\text{ mod }2$$

This repository will contain code that computes the average delay of Mesh-Collatz sequences, and records other interesting statistics.
