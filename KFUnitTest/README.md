# KFUnitTest

[KFUnitTest] is a framework for evaluating the reconstruction performance of the [KFParticle package](https://github.com/KFParticleGroup) using Monte Carlo data. It produces ROOT trees, QA histograms, and optional interactive plots for inspecting reconstructed particle quantities, pulls, and residuals.

KFUnitTest is primarily an integration test. The generated plots must be inspected to assess reconstruction quality; producing output files alone does not establish that the test has passed.

## Requirements

- [ROOT](https://root.cern/). The documented development and test environment uses **ROOT 6.26.04**. Compatibility with other versions is not established here.
- [KFParticle](https://github.com/KFParticleGroup), built from the branch you want to test.
- CMake, Make, and a C++ compiler compatible with your ROOT and KFParticle installations.
- [Geant4](https://geant4.web.cern.ch/) only if you want to generate your own Monte Carlo datasets.
- [RootInteractive](https://pypi.org/project/RootInteractive/) only if you want to generate interactive plots.

## Installation

### 1. Install ROOT

Follow the [ROOT installation instructions](https://root.cern/install/) and activate the ROOT environment for your installation before proceeding.

### 2. Build and install KFParticle

 Download the neccessary branch of **KFParticle** from one of the repositories to folder `<path-to-your-KF-folder/your-KF-folder>`
 Go to `<your-KF-folder>` and build the package via

        $ cd your-KF-folder
        $ mkdir build
        $ cd build
        $ cmake .. -DCMAKE_INSTALL_PREFIX=<path-to-your-KF-folder>/install
        $ make -j
        $ make install

Run the configuration, build, and installation commands from the **build directory**. `CMAKE_INSTALL_PREFIX` specifies where KFParticle will be installed, and `make -j` builds it in parallel.

After installation, set `KFPARTICLE_DIR` to the same installation prefix so that KFUnitTest can locate the KFParticle headers and libraries:

```bash
export KFPARTICLE_DIR="<path-to-your-KF-folder>/KFParticle/"
```

Set this variable in every terminal used to run KFUnitTest, or add the export command with your actual path to `~/.bashrc` if you use Bash.

### 3. Download KFUnitTest

### 4. (OPTIONAL) Prepare Monte Carlo data

Choose one of the following options:

- **Use an existing dataset.** Go to `MCData/<dataset-name>/` and run the corresponding `download-data-<dataset-name>.sh` script with Bash.
- **Generate your own dataset.** Install Geant4, go to `GeantPrograms/`, and run the corresponding `download-program-<dataset-name>.sh` script. Then build and run the downloaded program as described in `GeantPrograms/README.md`.

Replace `<dataset-name>` with the name of an available dataset. Run each download script from its containing directory.

## Configuration

Edit `UnitTests/ConfigConstants.h` before running a test. This file controls:

- The KFParticle version used by the test.
- Reconstruction modes and constraints.
- PDG checks.
- Input and output file names.

Ensure that the selected options match your KFParticle build and that the input paths point to the downloaded or generated Monte Carlo data.

## Running the test

With ROOT configured, `KFPARTICLE_DIR` set, and Monte Carlo data available, run the main macro from `UnitTests/`:

```bash
# From the KFUnitTest repository root:
cd UnitTests
root RunKFUnitTestMacro.cxx
```

The macro produces two ROOT files, with names determined by the configuration:

| Output file | Contents |
| --- | --- |
| `outputKFTrees_<data>.root` | Trees containing KFParticle structures before reconstruction (KFBR) and after reconstruction (KFAR). |
| `outputKFHistos_<data>.root` | QA histograms and plots, including pulls and residuals of reconstructed particle quantities. |

Inspect these outputs to evaluate the reconstruction performance of the KFParticle version under test.

## Interactive analysis

Install [RootInteractive](https://pypi.org/project/RootInteractive/) in the environment used for interactive analysis. Prepare the Monte Carlo data and configure `UnitTests/ConfigConstants.h` as described above.

Run the interactive macro from `InteractiveTest/`:

```bash
# From the KFUnitTest repository root:
cd InteractiveTest
root CreateInteractiveTest.cxx
```

The script generates:

| Output file | Contents |
| --- | --- |
| `InterTest_pulls.html` | Interactive pull plots. |
| `InterTest_residuals.html` | Interactive residual plots. |

Open these files in a web browser to explore the results.

## Adding custom histograms and plots

Use `OutputObjInterface` to add custom output objects:

1. Create a class derived from `OutputObjInterface`, located in `UnitTests/OutputObjInterface/`.
2. Use `ExampleOutput.h` in that directory as an implementation example.
3. Register an instance of your class in the `OutputMap` structure in `OutputMap.h`, following the existing example.
4. Run the test to generate your custom output.

## Workflow

1. Install ROOT and build the KFParticle branch to be evaluated.
2. Download or generate Monte Carlo data.
3. Configure the reconstruction options and file paths in `UnitTests/ConfigConstants.h`.
4. Run `RunKFUnitTestMacro.cxx` to generate ROOT trees and QA histograms.
5. Inspect the output to assess reconstruction quality.
6. Optionally run `CreateInteractiveTest.cxx` to generate browser-based pull and residual plots.
