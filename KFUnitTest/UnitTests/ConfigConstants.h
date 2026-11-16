#include <array>
#include "TString.h"
#pragma once

#include <vector>

// header file with function std::vector<float> MakeCovMatrix(const
// MCParticleStruct& part){ .... } to produce covariance matrix for particle
// Leave only one of the files uncommented
#include "MakeCovMat/MakeCovMatALICE.h"
// #include ...

// header file with function std::vector<float> MakeVertexCovMatrix(const
// MCParticleStruct& part){ .... } to produce covariance matrix for particles'
// vertexes Leave only one of the files uncommented If there is no uncommented
// headers, topological constraint for mother will be set according to MC vertex
// .
#include "MakeCovMat/MakeVertexCovMatALICE.h"
// #include ...

// Settings of KFParticle version
//#define TESTSUITE  // uncomment if you are using Testsuite branch of
// KFParticle #define ALIPHYSICS   // uncomment if you are using KFParticle
// inside AliPhysics

//#define CASCADES // for cascades - will be done later.... (?)
#define HomogeneousField
// const int NUM_OF_DAUGHTERS = 2;
// const int NUM_OF_MOTHERS = 1;
// const int NUM_OF_DECAYS = NUM_OF_MOTHERS;
// const int NUM_OF_KFPARTICLES = NUM_OF_MOTHERS + NUM_OF_DAUGHTERS;
const int MAX_NUM_OF_TRACKS = 50;

const int D0_PDG     = 421;
const int KMINUS_PDG = -321;
const int PIPLUS_PDG = 211;

const int JPSI_PDG   = 443;
const int eMINUS_PDG = 11;
const int ePLUS_PDG  = -11;

const int Xi_PDG      = 3312;
const int LAMBDA_PDG  = 3122;
const int PIMINUS_PDG = -211;
const int PROTON_PDG  = 2212;

const int NUM_OF_DECAYS                                                                 = 2;
const int MAX_NUM_OF_DAUGHTERS                                                          = 2;
const std::array<int, NUM_OF_DECAYS> MOTHER_PDG_IDS                                     = {{Xi_PDG, LAMBDA_PDG}};
const std::array<std::array<int, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> DAUGHTER_PDG_IDS = {
  {{LAMBDA_PDG, PIMINUS_PDG}, {PROTON_PDG, PIMINUS_PDG}}};
const std::array<int, NUM_OF_DECAYS> NUM_OF_DAUGHTERS = {{2, 2}};

// const int MOTHER_PDG = D0_PDG;
// const std::vector<int> DAUGTHERS_PDGS = {KMINUS_PDG, PIPLUS_PDG};

// const int MOTHER_PDG = JPSI_PDG;
// const std::vector<int> DAUGTHERS_PDGS = {eMINUS_PDG, ePLUS_PDG};

// const bool CHECK_PDG_MOTHER = true;
// const bool CHECK_PDG_DAUGHTERS = true;

// Method of reconstruction
const int CONSTRUCT_METHOD_NUMBER = 2;  // 0 or 2

// Mass and topological constraints
const bool SET_MASS_CONSTRAINT_MOTHER    = false;  //
const bool SET_MASS_CONSTRAINT_DAUGHTERS = true;   // set up nonlinear mass constraint

const bool SET_TOPOLOGICAL_CONSTRAINT_MOTHER    = true;  // must know covmat of primary vertex
const bool SET_TOPOLOGICAL_CONSTRAINT_DAUGHTERS = true;  //

// Settings of input and output files
// File with MC data and names of trees
const TString MCInputFile = "../MCData/Lambda0_to_Piminus_P/Lambda0_to_Piminus_P.root";
//../MCData/Xi_cascade/Xi_cascade.root";
//"../MCData/D0_to_Kminus_Piplus/D0_to_Kminus_Piplus.root";
const TString nameOfMCTree       = "Events";
const TString nameOfMagFieldTree = "MagField";
// Name of output file with KFParticles and MC output trees
const TString KFTreeOutputFile                                        = "outputKFTrees_D0_decay.root";
const std::array<TString, NUM_OF_DECAYS> nameOfMCInitialOutputTree    = {{"MCInitial_Xi", "MCInitial_Lambda"}};
const std::array<TString, NUM_OF_DECAYS> nameOfMCFinalOutputTree      = {{"MCFinal_Xi", "MCFinal_Lambda"}};
const std::array<TString, NUM_OF_DECAYS> nameOfKFProductionVertexTree = {{"KFBeforeReco_Xi", "KFBeforeReco_Lambda"}};
const std::array<TString, NUM_OF_DECAYS> nameOfKFDecayVertexTree      = {{"KFAfterReco_Xi", "KFAfterReco_Lambda"}};

// Name of output file with QA plots and histograms
const TString KFInputFile       = KFTreeOutputFile;
const TString KFHistoOutputFile = "outputKFHistos_D0_decay.root";
