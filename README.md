# SEPAR  

SEPAR (Spatial gene Expression PAttern Recognition) is a computational method for analyzing spatial transcriptomics data.  

## Installation  

Download SEPAR code from this repository and ensure you have the following dependencies:  

```python  
pandas>=2.0.3  
numpy>=1.23.5  
scanpy>=1.9.6  
anndata>=0.8.0  
matplotlib>=3.6.1  
scipy>=1.10.0  
scikit-learn>=1.2.0  
tqdm>=4.64.1
```

## Quick Start

```
import scanpy as sc  
from separ import SEPAR  

# Load data  
adata = sc.read_h5ad('your_data.h5ad')  

# Initialize and run SEPAR  
separ = SEPAR(adata, n_cluster=8)  
separ.preprocess()  
separ.compute_graph()  
separ.compute_weight()  
separ.separ_algorithm(r=30, alpha=1.0, beta=0.1, gamma=0.1)  
separ.clustering()  

```


