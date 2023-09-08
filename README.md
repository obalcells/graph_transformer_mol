Final task in Introduction to Machine Learning Course: Discovering Future Solar Energy Materials

We use graph transformers and CNNs to calculate the energy gap between the most and least excited states of a given molecule which is given simply as the SMILES representation. We process the molecule and convert it into a graph where nodes are atoms and edges are bonds.

The first token in the context we feed to the transformer represents the whole molecule, so it encodes information about the entire structure of the molecule. The 2-dimensional and 3-dimensional graph information is encoded into the attention pattern. See https://arxiv.org/pdf/2210.01765.pdf for more details on the model architecture.

The model achieves a RMSE of 0.07eV on the hidden test data, which matches SOTA results.
