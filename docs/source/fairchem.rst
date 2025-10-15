FAIRChem Datasets
=================

The FAIRChem scraper enables FitSNAP to read and process datasets from Meta's FAIR Chemistry project, including large-scale catalysis and materials datasets stored in LMDB format.

Overview
--------

The FAIRChem scraper (``fairchem_scraper.py``) provides read-only access to ASE-compatible LMDB datasets with built-in MPI parallelization for efficient processing of large datasets. It supports multiple concurrent datasets and handles automatic element filtering and configuration distribution across compute ranks.

Supported Datasets
------------------

The scraper supports all FAIRChem datasets that use the LMDB format:

**Catalysis Datasets**

**Open Catalyst 2020 (OC20)**
   - **Description**: 1.3M+ DFT calculations for catalyst discovery
   - **Systems**: Adsorbates on catalyst surfaces
   - **Download**: https://fair-chem.github.io/catalysts/datasets/oc20.html
   - **Paper**: https://arxiv.org/abs/2010.09990

**Open Catalyst 2022 (OC22)**
   - **Description**: 62k+ DFT calculations with enhanced adsorbate coverage
   - **Systems**: Expanded adsorbate-catalyst combinations  
   - **Download**: https://fair-chem.github.io/catalysts/datasets/oc22.html
   - **Paper**: https://arxiv.org/abs/2206.08917

**Materials Datasets**

**Open Materials 2024 (OMat24)**
   - **Description**: 110M+ structures across diverse inorganic material classes
   - **Systems**: Bulk crystals, surfaces, non-equilibrium structures
   - **Download**: https://fair-chem.github.io/core/datasets/omat24.html
   - **Paper**: https://arxiv.org/abs/2410.12771

**Materials Project Trajectories (MPtrj)**
   - **Description**: 1.5M+ structures from Materials Project DFT relaxations
   - **Systems**: Inorganic crystals with forces, energies, and magnetic moments
   - **Download**: https://figshare.com/articles/dataset/Materials_Project_Trjectory_MPtrj_Dataset/23713842
   - **Paper**: https://doi.org/10.1038/s42256-023-00716-3

**sAlex (Subsampled Alexandria)**
   - **Description**: Matbench-Discovery compliant subset of Alexandria dataset
   - **Systems**: Materials trajectory data filtered for redundancy
   - **Download**: Available with OMat24 at https://huggingface.co/datasets/facebook/OMAT24
   - **Paper**: https://arxiv.org/abs/2410.12771

**Molecular Datasets**

**Open Molecules 2025 (OMol25)**
   - **Description**: 100M+ high-accuracy molecular DFT calculations
   - **Systems**: Small molecules, biomolecules, metal complexes, electrolytes (up to 350 atoms)
   - **Level of Theory**: ωB97M-V/def2-TZVPD
   - **Download**: https://fair-chem.github.io/molecules/datasets/omol25.html
   - **Paper**: https://arxiv.org/abs/2505.08762

**Direct Air Capture Datasets**

**Open Direct Air Capture 2023 (ODAC23)** [Deprecated]
   - **Description**: DFT calculations for CO₂ capture in MOFs
   - **Systems**: Metal-organic frameworks for direct air capture
   - **Download**: https://fair-chem.github.io/dac/datasets/odac23.html
   - **Paper**: Available on FAIRChem site

**Open Direct Air Capture 2025 (ODAC25)**
   - **Description**: 70M+ DFT calculations for CO₂, H₂O, N₂, O₂ adsorption
   - **Systems**: 15,000+ metal-organic frameworks with diverse functionalization
   - **Download**: https://fair-chem.github.io/dac/datasets/odac25.html
   - **Paper**: Available on FAIRChem site

**Custom Datasets**
   - Any ASE-compatible LMDB dataset following FAIRChem conventions
   - Custom datasets can be processed if they follow the same structure

Installation Requirements
-------------------------

The FAIRChem scraper requires additional dependency:

.. code-block:: bash

   pip install fairchem-core

Or for development installations:

.. code-block:: bash

   pip install -e ".[fairchem]"

Configuration
-------------

Basic Setup
~~~~~~~~~~~

Configure the scraper in your FitSNAP input file:

.. code-block:: ini

   [SCRAPER]
   scraper_name = fairchem

   [PATH]
   datapath = /path/to/fairchem/dataset

   [GROUPS]
   group_sections = group1 group2
   group_types = train train
   smartweights = False
   
   [group1]
   group_sections = group1
   training_size = 1.0
   testing_size = 0.0
   eweight = 1.0
   fweight = 1.0
   vweight = 0.0

   [group2]
   group_sections = group2  
   training_size = 1.0
   testing_size = 0.0
   eweight = 1.0
   fweight = 1.0
   vweight = 0.0

Multi-Dataset Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each group name corresponds to a subdirectory containing an LMDB dataset:

.. code-block:: text

   /path/to/fairchem/datasets/
   ├── oc20_train/          # group1 -> oc20_train subdataset
   │   ├── data.lmdb
   │   └── lock.mdb
   ├── oc22_train/          # group2 -> oc22_train subdataset  
   │   ├── data.lmdb
   │   └── lock.mdb
   └── omat24_train/        # group3 -> omat24_train subdataset
       ├── data.lmdb
       └── lock.mdb

Advanced Options
~~~~~~~~~~~~~~~~

Additional scraper-specific options:

.. code-block:: ini

   [SCRAPER]
   scraper_name = fairchem
   max_configs_per_rank = 10000    # Limit configurations per MPI rank
   require_energy = True           # Skip configs without energy
   require_forces = True           # Skip configs without forces  
   verbose = False                 # Enable detailed logging

Element Filtering
~~~~~~~~~~~~~~~~~

The scraper automatically filters configurations based on allowed elements from your potential configuration:

.. code-block:: ini

   [BISPECTRUM]
   types = Al Cu O H          # Only configs with these elements will be processed

   [ACE] 
   types = C H N O            # Alternative element specification

Usage Examples
--------------

Single Dataset
~~~~~~~~~~~~~~

Process a single OC20 training dataset:

.. code-block:: ini

   [PATH]
   datapath = /data/oc20

   [GROUPS]
   group_sections = oc20_train
   group_types = train
   
   [oc20_train]
   group_sections = oc20_train
   training_size = 1.0
   eweight = 1.0
   fweight = 1.0

Multiple Datasets
~~~~~~~~~~~~~~~~~

Combine multiple datasets with different weights:

.. code-block:: ini

   [PATH] 
   datapath = /data/fairchem

   [GROUPS]
   group_sections = oc20_train oc22_train omat24_bulk
   group_types = train train train

   [oc20_train]
   group_sections = oc20_train
   training_size = 0.8
   testing_size = 0.2
   eweight = 1.0
   fweight = 1.0

   [oc22_train] 
   group_sections = oc22_train
   training_size = 1.0
   eweight = 2.0      # Higher energy weight
   fweight = 0.5      # Lower force weight

   [omat24_bulk]
   group_sections = omat24_bulk
   training_size = 1.0
   eweight = 1.0
   fweight = 1.0

Large-Scale Processing
~~~~~~~~~~~~~~~~~~~~~~

For large datasets, use MPI parallelization and memory management:

.. code-block:: bash

   # Run with MPI across multiple nodes
   mpirun -np 64 python -m fitsnap3 input.in

.. code-block:: ini

   [SCRAPER]
   scraper_name = fairchem
   max_configs_per_rank = 5000     # Limit memory usage per rank
   require_energy = True
   require_forces = True

Performance Considerations
--------------------------

**Memory Management**
   - Use ``max_configs_per_rank`` to control memory usage per MPI rank
   - Large datasets benefit from distributed processing across multiple nodes

**I/O Optimization**
   - LMDB provides efficient random access to configurations
   - The scraper uses ``MDB_NOLOCK`` for distributed filesystem compatibility
   - Each MPI rank processes a disjoint subset of configurations

**Element Filtering**
   - Filtering by allowed elements happens early to reduce processing overhead
   - Configure ``types`` in your potential section to match your system of interest

Data Format
-----------

The scraper extracts the following information from each configuration:

**Required Fields**
   - ``positions``: Atomic coordinates (Å)
   - ``cell``: Unit cell vectors (Å) 
   - ``symbols``: Element symbols
   - ``energy``: Total energy (eV)

**Optional Fields**
   - ``forces``: Atomic forces (eV/Å)
   - ``stress``: Stress tensor (eV/Å³)

**Metadata**
   - ``Group``: Dataset/group identifier
   - ``File``: Configuration identifier
   - ``NumAtoms``: Number of atoms
   - ``eweight/fweight``: Energy/force weights

Error Handling
--------------

The scraper includes robust error handling:

- **Missing Dependencies**: Clear error messages for missing ``lmdb`` or ``fairchem-core``
- **Invalid Configurations**: Automatic skipping of corrupted or incomplete structures
- **Element Filtering**: Silent filtering of configurations with disallowed elements
- **MPI Safety**: Proper synchronization and error propagation across ranks

Troubleshooting
---------------

**Import Errors**
   Ensure ``lmdb`` and ``fairchem-core`` are installed:
   
   .. code-block:: bash
   
      pip install lmdb fairchem-core

**Memory Issues**
   Reduce memory usage per rank:
   
   .. code-block:: ini
   
      [SCRAPER]
      max_configs_per_rank = 1000

**Empty Datasets**
   Check element filtering and dataset paths:
   
   .. code-block:: ini
   
      [SCRAPER]
      verbose = True    # Enable detailed logging

**Distributed Filesystem Issues**
   The scraper automatically uses ``MDB_NOLOCK`` for compatibility with NFS and similar filesystems.

Related Documentation
---------------------

- **FAIRChem Project**: https://github.com/Open-Catalyst-Project/ocp
- **Dataset Downloads**: https://github.com/Open-Catalyst-Project/ocp/blob/main/DATASET.md  
- **LMDB Documentation**: https://lmdb.readthedocs.io/
- **ASE Documentation**: https://wiki.fysik.dtu.dk/ase/
- **FitSNAP Scrapers**: :doc:`/Lib/scraper`
- **MPI Usage**: :doc:`/Run/parallel`

Citations
---------

If you use FAIRChem datasets in your research, please cite the appropriate papers:

**OC20:**
   Chanussot, L., Das, A., Goyal, S. et al. Open Catalyst 2020 (OC20) Dataset and Community Challenges for Catalysis. *ACS Catal.* 2021, 11, 6059-6072.

**OC22:**  
   Tran, R., Lan, J., Shuaibi, M. et al. The Open Catalyst 2022 (OC22) Dataset and Challenges for Oxide Electrocatalysis. *ACS Catal.* 2023, 13, 3066-3084.

**OMat24:**
   Barroso-Luque, L., Shuaibi, M., Fu, X. et al. Open Materials 2024 (OMat24) Inorganic Materials Dataset and Models. *arXiv preprint* 2024, arXiv:2410.12771.

**MPtrj:**
   Deng, B., Zhong, P., Jun, K. et al. CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling. *Nat. Mach. Intell.* 2023, 5, 1031-1041.

**OMol25:**
   Levine, D.S., Shuaibi, M., Spotte-Smith, E.W.C. et al. The Open Molecules 2025 (OMol25) Dataset, Evaluations, and Models. *arXiv preprint* 2025, arXiv:2505.08762.

**ODAC23/ODAC25:**
   Sriram, A., Choi, S., Yu, X. et al. Open Direct Air Capture datasets for CO₂ capture in metal-organic frameworks. Available on FAIRChem documentation.

**Alexandria/sAlex:**
   Schmidt, J., Hoffmann, N., Wang, H.-C. et al. Machine-Learning-Assisted Determination of the Global Zero-Temperature Phase Diagram of Materials. *Adv. Mater.* 2023, 35, 2210788.
