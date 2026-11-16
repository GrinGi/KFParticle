#include "ConfigConstants.h"
#include "HeadersList.h"
#include "KFParticle/KFParticle.h"
#include "ParticleStructures/MCParticleStructure.h"
#include "TFile.h"
#include "TString.h"
#include "TTree.h"

#include <iostream>


void MakeKFParticleTrees()
{

  cout << "\n Starting MakeKFParticleTrees macro \n";

  std::cout << "Reconstruction settings settings:\n";
  std::cout << "  - Construct method = " << CONSTRUCT_METHOD_NUMBER << "\n";
  std::cout << "  - Mother Mass constraint = " << SET_MASS_CONSTRAINT_MOTHER << "\n";
  std::cout << "  - Daughters mass constraint = " << SET_MASS_CONSTRAINT_DAUGHTERS << "\n";
  std::cout << "  - Mother topological constraint = " << SET_TOPOLOGICAL_CONSTRAINT_MOTHER << "\n";
  std::cout << "  - Daughters topological constraint = " << SET_TOPOLOGICAL_CONSTRAINT_DAUGHTERS << "\n";

#ifndef MAKE_VERTEX_COV_MAT
  if (SET_TOPOLOGICAL_CONSTRAINT_MOTHER) {
    std::cout << "\nNo method to smear primary vertex is provided. Smearing will be done according to MC mother track "
                 "vertex.\n\n";
  }
#endif

  //get magnetic field from file......

#ifdef HomogeneousField  //ALICE
  std::cout << "HomogeneousField option is set" << std::endl;
#endif
#ifdef NonhomogeneousField  //cbm
  std::cout << "NonhomogeneousField option is set" << std::endl;
#endif

  TChain chainMagField(nameOfMagFieldTree);
  std::cout << "\nMCInputFile: " << MCInputFile << '\n' << std::endl;
  chainMagField.Add(MCInputFile);

  float magField[3];
  chainMagField.SetBranchAddress("magField", &magField);
  chainMagField.GetEntry(0);
  std::cout << "Magnetic Field {Bx,By,Bz} = " << magField[0] << "   " << magField[1] << "   " << magField[2]
            << "   tesla" << std::endl;

  // check the field
  //KFParticle::SetField(magField[2] * 10);  // Bz, *10 to convert tesla to kgauss
  {
//    float xyz[3] = {0, 0, 0};
  //  float B[3];
   // KFParticle p;
   // p.GetFieldValue(xyz, B);
   // std::cout << "Field is set to " << B[0] << " " << B[1] << " " << B[2] << " kilogaus" << std::endl;
  }

  ///////////////////////////////////////////////////////
  // Lets read data from the input file and create output file
  //ROOT file reading...
  ///////////////////////////////////////////////////////
  cout << "Reading input and createing output\n";
  TChain chain(nameOfMCTree);
  std::cout << "\nMCInputFile: " << MCInputFile << '\n' << std::endl;
  chain.Add(MCInputFile);

  int nTracks_MC;
  int trackID_MC[MAX_NUM_OF_TRACKS];
  int parentID_MC[MAX_NUM_OF_TRACKS];
  int pdg_MC[MAX_NUM_OF_TRACKS];
  //float pdgLifeTime_MC[MAX_NUM_OF_TRACKS];
  int charge_MC[MAX_NUM_OF_TRACKS];
  float initialX_MC[MAX_NUM_OF_TRACKS];
  float initialY_MC[MAX_NUM_OF_TRACKS];
  float initialZ_MC[MAX_NUM_OF_TRACKS];
  float finalX_MC[MAX_NUM_OF_TRACKS];
  float finalY_MC[MAX_NUM_OF_TRACKS];
  float finalZ_MC[MAX_NUM_OF_TRACKS];
  float trackLength_MC[MAX_NUM_OF_TRACKS];
  float initialPX_MC[MAX_NUM_OF_TRACKS];
  float initialPY_MC[MAX_NUM_OF_TRACKS];
  float initialPZ_MC[MAX_NUM_OF_TRACKS];
  float finalPX_MC[MAX_NUM_OF_TRACKS];
  float finalPY_MC[MAX_NUM_OF_TRACKS];
  float finalPZ_MC[MAX_NUM_OF_TRACKS];
  float mass_MC[MAX_NUM_OF_TRACKS];

  chain.SetBranchAddress("nTracks", &nTracks_MC);
  chain.SetBranchAddress("trackID", trackID_MC);
  chain.SetBranchAddress("parentID", parentID_MC);
  chain.SetBranchAddress("pdg", pdg_MC);
  //chain.SetBranchAddress( "pdgLifeTime", pdgLifeTime_MC);
  chain.SetBranchAddress("charge", charge_MC);
  chain.SetBranchAddress("initialX", initialX_MC);
  chain.SetBranchAddress("initialY", initialY_MC);
  chain.SetBranchAddress("initialZ", initialZ_MC);
  chain.SetBranchAddress("finalX", finalX_MC);
  chain.SetBranchAddress("finalY", finalY_MC);
  chain.SetBranchAddress("finalZ", finalZ_MC);
  chain.SetBranchAddress("trackLength", trackLength_MC);
  chain.SetBranchAddress("initialPX", initialPX_MC);
  chain.SetBranchAddress("initialPY", initialPY_MC);
  chain.SetBranchAddress("initialPZ", initialPZ_MC);
  chain.SetBranchAddress("finalPX", finalPX_MC);
  chain.SetBranchAddress("finalPY", finalPY_MC);
  chain.SetBranchAddress("finalPZ", finalPZ_MC);
  chain.SetBranchAddress("mass", mass_MC);

  // output creating
  TFile outputFile(KFTreeOutputFile, "RECREATE");

  // ##### prepare MCOutputtree
  TTree treeMCInitial[NUM_OF_DECAYS];  // = new TTree("KFBeforeReco", "KFBeforeReco");
  for (int iDecay = 0; iDecay < NUM_OF_DECAYS; iDecay++) {
    treeMCInitial[iDecay].SetObject(nameOfMCInitialOutputTree[iDecay], nameOfMCInitialOutputTree[iDecay]);
  }

  int nTracks_MCInitial[NUM_OF_DECAYS];
  int numOfMCEvent_MCInitial[NUM_OF_DECAYS];

  int id_MCInitial[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  int parentid_MCInitial[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  int pdg_MCInitial[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float x_MCInitial[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float y_MCInitial[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float z_MCInitial[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float pX_MCInitial[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float pY_MCInitial[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float pZ_MCInitial[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float e_MCInitial[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float s_MCInitial[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  int charge_MCInitial[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];

  bool isMother_MCInitial[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  bool isDaughter_MCInitial[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];

  float p_MCInitial[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float pT_MCInitial[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float eta_MCInitial[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float phi_MCInitial[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float mass_MCInitial[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float decayLength_MCInitial[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float decayLengthXY_MCInitial[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float lifeTime_MCInitial[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float r_MCInitial[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float rapidity_MCInitial[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float theta_MCInitial[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];

  for (int iDecay = 0; iDecay < NUM_OF_DECAYS; iDecay++) {
    treeMCInitial[iDecay].Branch("nTracks", &nTracks_MCInitial[iDecay], "nTracks/I");
    treeMCInitial[iDecay].Branch("numOfMCEvent", &numOfMCEvent_MCInitial[iDecay], "numOfMCEvent/I");

    treeMCInitial[iDecay].Branch("ID", id_MCInitial[iDecay], "ID[nTracks]/I");
    treeMCInitial[iDecay].Branch("ParentID", parentid_MCInitial[iDecay], "ParentID[nTracks]/I");
    treeMCInitial[iDecay].Branch("pdgID", pdg_MCInitial[iDecay], "pdgID[nTracks]/I");
    treeMCInitial[iDecay].Branch("x", &x_MCInitial[iDecay], "x[nTracks]/F");
    treeMCInitial[iDecay].Branch("y", &y_MCInitial[iDecay], "y[nTracks]/F");
    treeMCInitial[iDecay].Branch("z", &z_MCInitial[iDecay], "z[nTracks]/F");
    treeMCInitial[iDecay].Branch("pX", &pX_MCInitial[iDecay], "pX[nTracks]/F");
    treeMCInitial[iDecay].Branch("pY", &pY_MCInitial[iDecay], "pY[nTracks]/F");
    treeMCInitial[iDecay].Branch("pZ", &pZ_MCInitial[iDecay], "pZ[nTracks]/F");
    treeMCInitial[iDecay].Branch("e", &e_MCInitial[iDecay], "e[nTracks]/F");
    treeMCInitial[iDecay].Branch("s", &s_MCInitial[iDecay], "s[nTracks]/F");
    treeMCInitial[iDecay].Branch("charge", &charge_MCInitial[iDecay], "charge[nTracks]/I");

    treeMCInitial[iDecay].Branch("isMother", &isMother_MCInitial[iDecay], "isMother[nTracks]/O");
    treeMCInitial[iDecay].Branch("isDaughter", &isDaughter_MCInitial[iDecay], "isDaughter[nTracks]/O");

    treeMCInitial[iDecay].Branch("p", &p_MCInitial[iDecay], "p[nTracks]/F");
    treeMCInitial[iDecay].Branch("pT", &pT_MCInitial[iDecay], "pT[nTracks]/F");
    treeMCInitial[iDecay].Branch("eta", &eta_MCInitial[iDecay], "eta[nTracks]/F");
    treeMCInitial[iDecay].Branch("phi", &phi_MCInitial[iDecay], "phi[nTracks]/F");
    treeMCInitial[iDecay].Branch("mass", &mass_MCInitial[iDecay], "mass[nTracks]/F");
    treeMCInitial[iDecay].Branch("decayLength", &decayLength_MCInitial[iDecay], "decayLength[nTracks]/F");
    treeMCInitial[iDecay].Branch("decayLengthXY", &decayLengthXY_MCInitial[iDecay], "decayLengthXY[nTracks]/F");
    treeMCInitial[iDecay].Branch("lifeTime", &lifeTime_MCInitial[iDecay], "lifeTime[nTracks]/F");
    treeMCInitial[iDecay].Branch("r", &r_MCInitial[iDecay], "r[nTracks]/F");
    treeMCInitial[iDecay].Branch("rapidity", &rapidity_MCInitial[iDecay], "rapidity[nTracks]/F");
    treeMCInitial[iDecay].Branch("theta", &theta_MCInitial[iDecay], "theta[nTracks]/F");
  }

  TTree treeMCFinal[NUM_OF_DECAYS];  // = new TTree("KFBeforeReco", "KFBeforeReco");
  for (int iDecay = 0; iDecay < NUM_OF_DECAYS; iDecay++) {
    treeMCFinal[iDecay].SetObject(nameOfMCFinalOutputTree[iDecay], nameOfMCFinalOutputTree[iDecay]);
  }

  int nTracks_MCFinal[NUM_OF_DECAYS];
  int numOfMCEvent_MCFinal[NUM_OF_DECAYS];

  int id_MCFinal[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  int parentid_MCFinal[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  int pdg_MCFinal[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float x_MCFinal[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float y_MCFinal[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float z_MCFinal[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float pX_MCFinal[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float pY_MCFinal[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float pZ_MCFinal[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float e_MCFinal[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float s_MCFinal[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  int charge_MCFinal[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];

  bool isMother_MCFinal[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  bool isDaughter_MCFinal[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];

  float p_MCFinal[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float pT_MCFinal[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float eta_MCFinal[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float phi_MCFinal[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float mass_MCFinal[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float decayLength_MCFinal[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float decayLengthXY_MCFinal[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float lifeTime_MCFinal[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float r_MCFinal[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float rapidity_MCFinal[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float theta_MCFinal[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];

  for (int iDecay = 0; iDecay < NUM_OF_DECAYS; iDecay++) {
    treeMCFinal[iDecay].Branch("nTracks", &nTracks_MCFinal[iDecay], "nTracks/I");
    treeMCFinal[iDecay].Branch("numOfMCEvent", &numOfMCEvent_MCFinal[iDecay], "numOfMCEvent/I");

    treeMCFinal[iDecay].Branch("ID", id_MCFinal[iDecay], "ID[nTracks]/I");
    treeMCFinal[iDecay].Branch("ParentID", parentid_MCFinal[iDecay], "ParentID[nTracks]/I");
    treeMCFinal[iDecay].Branch("pdgID", pdg_MCFinal[iDecay], "pdgID[nTracks]/I");
    treeMCFinal[iDecay].Branch("x", &x_MCFinal[iDecay], "x[nTracks]/F");
    treeMCFinal[iDecay].Branch("y", &y_MCFinal[iDecay], "y[nTracks]/F");
    treeMCFinal[iDecay].Branch("z", &z_MCFinal[iDecay], "z[nTracks]/F");
    treeMCFinal[iDecay].Branch("pX", &pX_MCFinal[iDecay], "pX[nTracks]/F");
    treeMCFinal[iDecay].Branch("pY", &pY_MCFinal[iDecay], "pY[nTracks]/F");
    treeMCFinal[iDecay].Branch("pZ", &pZ_MCFinal[iDecay], "pZ[nTracks]/F");
    treeMCFinal[iDecay].Branch("e", &e_MCFinal[iDecay], "e[nTracks]/F");
    treeMCFinal[iDecay].Branch("s", &s_MCFinal[iDecay], "s[nTracks]/F");
    treeMCFinal[iDecay].Branch("charge", &charge_MCFinal[iDecay], "charge[nTracks]/I");

    treeMCFinal[iDecay].Branch("isMother", &isMother_MCFinal[iDecay], "isMother[nTracks]/O");
    treeMCFinal[iDecay].Branch("isDaughter", &isDaughter_MCFinal[iDecay], "isDaughter[nTracks]/O");

    treeMCFinal[iDecay].Branch("p", &p_MCFinal[iDecay], "p[nTracks]/F");
    treeMCFinal[iDecay].Branch("pT", &pT_MCFinal[iDecay], "pT[nTracks]/F");
    treeMCFinal[iDecay].Branch("eta", &eta_MCFinal[iDecay], "eta[nTracks]/F");
    treeMCFinal[iDecay].Branch("phi", &phi_MCFinal[iDecay], "phi[nTracks]/F");
    treeMCFinal[iDecay].Branch("mass", &mass_MCFinal[iDecay], "mass[nTracks]/F");
    treeMCFinal[iDecay].Branch("decayLength", &decayLength_MCFinal[iDecay], "decayLength[nTracks]/F");
    treeMCFinal[iDecay].Branch("decayLengthXY", &decayLengthXY_MCFinal[iDecay], "decayLengthXY[nTracks]/F");
    treeMCFinal[iDecay].Branch("lifeTime", &lifeTime_MCFinal[iDecay], "lifeTime[nTracks]/F");
    treeMCFinal[iDecay].Branch("r", &r_MCFinal[iDecay], "r[nTracks]/F");
    treeMCFinal[iDecay].Branch("rapidity", &rapidity_MCFinal[iDecay], "rapidity[nTracks]/F");
    treeMCFinal[iDecay].Branch("theta", &theta_MCFinal[iDecay], "theta[nTracks]/F");
  }

  // ##### prepare KFProductionVertextree
  TTree treeKFProductionVertex[NUM_OF_DECAYS];  // = new TTree("KFBeforeReco", "KFBeforeReco");
  for (int iDecay = 0; iDecay < NUM_OF_DECAYS; iDecay++) {
    treeKFProductionVertex[iDecay].SetObject(nameOfKFProductionVertexTree[iDecay],
                                             nameOfKFProductionVertexTree[iDecay]);
  }

  int nTracks_KFProductionVertex[NUM_OF_DECAYS];
  int numOfMCEvent_KFProductionVertex[NUM_OF_DECAYS];
  bool isAllCovMatOK_KFProductionVertex[NUM_OF_DECAYS];

  int id_KFProductionVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  int idMC_KFProductionVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  int pdg_KFProductionVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float params_KFProductionVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS][6];
  float covMat_KFProductionVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS][36];
  bool isCovMatOK_KFProductionVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float x_KFProductionVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float y_KFProductionVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float z_KFProductionVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float pX_KFProductionVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float pY_KFProductionVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float pZ_KFProductionVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float e_KFProductionVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float s_KFProductionVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  int charge_KFProductionVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float chi2_KFProductionVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float ndf_KFProductionVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float massHypo_KFProductionVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];

  bool isMother_KFProductionVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  bool isDaughter_KFProductionVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];

  float p_KFProductionVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float pT_KFProductionVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float eta_KFProductionVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float phi_KFProductionVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float mass_KFProductionVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float decayLength_KFProductionVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float decayLengthXY_KFProductionVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float lifeTime_KFProductionVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float r_KFProductionVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float rapidity_KFProductionVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float theta_KFProductionVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];

  float xErr_KFProductionVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float yErr_KFProductionVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float zErr_KFProductionVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float pXErr_KFProductionVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float pYErr_KFProductionVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float pZErr_KFProductionVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float eErr_KFProductionVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float sErr_KFProductionVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float pErr_KFProductionVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float pTErr_KFProductionVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float etaErr_KFProductionVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float phiErr_KFProductionVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float massErr_KFProductionVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float decayLengthErr_KFProductionVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float decayLengthXYErr_KFProductionVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float lifeTimeErr_KFProductionVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float rErr_KFProductionVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];

  for (int iDecay = 0; iDecay < NUM_OF_DECAYS; iDecay++) {
    treeKFProductionVertex[iDecay].Branch("nTracks", &nTracks_KFProductionVertex[iDecay], "nTracks/I");
    treeKFProductionVertex[iDecay].Branch("numOfMCEvent", &numOfMCEvent_KFProductionVertex[iDecay], "numOfMCEvent/I");
    treeKFProductionVertex[iDecay].Branch("isAllCovMatOK", &isAllCovMatOK_KFProductionVertex[iDecay],
                                          "isAllCovMatOK/O");

    treeKFProductionVertex[iDecay].Branch("ID", id_KFProductionVertex[iDecay], "ID[nTracks]/I");
    treeKFProductionVertex[iDecay].Branch("IDMC", idMC_KFProductionVertex[iDecay], "IDMC[nTracks]/I");
    treeKFProductionVertex[iDecay].Branch("pdgID", pdg_KFProductionVertex[iDecay], "pdgID[nTracks]/I");
    treeKFProductionVertex[iDecay].Branch("params", &params_KFProductionVertex[iDecay], "params[nTracks][6]/F");
    treeKFProductionVertex[iDecay].Branch("covMat", &covMat_KFProductionVertex[iDecay], "covMat[nTracks][36]/F");
    treeKFProductionVertex[iDecay].Branch("isCovMatOK", &isCovMatOK_KFProductionVertex[iDecay],
                                          "isCovMatOK[nTracks]/O");
    treeKFProductionVertex[iDecay].Branch("x", &x_KFProductionVertex[iDecay], "x[nTracks]/F");
    treeKFProductionVertex[iDecay].Branch("y", &y_KFProductionVertex[iDecay], "y[nTracks]/F");
    treeKFProductionVertex[iDecay].Branch("z", &z_KFProductionVertex[iDecay], "z[nTracks]/F");
    treeKFProductionVertex[iDecay].Branch("pX", &pX_KFProductionVertex[iDecay], "pX[nTracks]/F");
    treeKFProductionVertex[iDecay].Branch("pY", &pY_KFProductionVertex[iDecay], "pY[nTracks]/F");
    treeKFProductionVertex[iDecay].Branch("pZ", &pZ_KFProductionVertex[iDecay], "pZ[nTracks]/F");
    treeKFProductionVertex[iDecay].Branch("e", &e_KFProductionVertex[iDecay], "e[nTracks]/F");
    treeKFProductionVertex[iDecay].Branch("s", &s_KFProductionVertex[iDecay], "s[nTracks]/F");
    treeKFProductionVertex[iDecay].Branch("charge", &charge_KFProductionVertex[iDecay], "charge[nTracks]/I");
    treeKFProductionVertex[iDecay].Branch("chi2", &chi2_KFProductionVertex[iDecay], "chi2[nTracks]/F");
    treeKFProductionVertex[iDecay].Branch("ndf", &ndf_KFProductionVertex[iDecay], "ndf[nTracks]/F");
    treeKFProductionVertex[iDecay].Branch("massHypo", &massHypo_KFProductionVertex[iDecay], "massHypo[nTracks]/F");

    treeKFProductionVertex[iDecay].Branch("isMother", &isMother_KFProductionVertex[iDecay], "isMother[nTracks]/O");
    treeKFProductionVertex[iDecay].Branch("isDaughter", &isDaughter_KFProductionVertex[iDecay],
                                          "isDaughter[nTracks]/O");

    treeKFProductionVertex[iDecay].Branch("p", &p_KFProductionVertex[iDecay], "p[nTracks]/F");
    treeKFProductionVertex[iDecay].Branch("pT", &pT_KFProductionVertex[iDecay], "pT[nTracks]/F");
    treeKFProductionVertex[iDecay].Branch("eta", &eta_KFProductionVertex[iDecay], "eta[nTracks]/F");
    treeKFProductionVertex[iDecay].Branch("phi", &phi_KFProductionVertex[iDecay], "phi[nTracks]/F");
    treeKFProductionVertex[iDecay].Branch("mass", &mass_KFProductionVertex[iDecay], "mass[nTracks]/F");
    treeKFProductionVertex[iDecay].Branch("decayLength", &decayLength_KFProductionVertex[iDecay],
                                          "decayLength[nTracks]/F");
    treeKFProductionVertex[iDecay].Branch("decayLengthXY", &decayLengthXY_KFProductionVertex[iDecay],
                                          "decayLengthXY[nTracks]/F");
    treeKFProductionVertex[iDecay].Branch("lifeTime", &lifeTime_KFProductionVertex[iDecay], "lifeTime[nTracks]/F");
    treeKFProductionVertex[iDecay].Branch("r", &r_KFProductionVertex[iDecay], "r[nTracks]/F");
    treeKFProductionVertex[iDecay].Branch("rapidity", &rapidity_KFProductionVertex[iDecay], "rapidity[nTracks]/F");
    treeKFProductionVertex[iDecay].Branch("theta", &theta_KFProductionVertex[iDecay], "theta[nTracks]/F");

    treeKFProductionVertex[iDecay].Branch("xErr", &xErr_KFProductionVertex[iDecay], "xErr[nTracks]/F");
    treeKFProductionVertex[iDecay].Branch("yErr", &yErr_KFProductionVertex[iDecay], "yErr[nTracks]/F");
    treeKFProductionVertex[iDecay].Branch("zErr", &zErr_KFProductionVertex[iDecay], "zErr[nTracks]/F");
    treeKFProductionVertex[iDecay].Branch("pXErr", &pXErr_KFProductionVertex[iDecay], "pXErr[nTracks]/F");
    treeKFProductionVertex[iDecay].Branch("pYErr", &pYErr_KFProductionVertex[iDecay], "pYErr[nTracks]/F");
    treeKFProductionVertex[iDecay].Branch("pZErr", &pZErr_KFProductionVertex[iDecay], "pZErr[nTracks]/F");
    treeKFProductionVertex[iDecay].Branch("eErr", &eErr_KFProductionVertex[iDecay], "eErr[nTracks]/F");
    treeKFProductionVertex[iDecay].Branch("sErr", &sErr_KFProductionVertex[iDecay], "sErr[nTracks]/F");
    treeKFProductionVertex[iDecay].Branch("pErr", &pErr_KFProductionVertex[iDecay], "pErr[nTracks]/F");
    treeKFProductionVertex[iDecay].Branch("pTErr", &pTErr_KFProductionVertex[iDecay], "pTErr[nTracks]/F");
    treeKFProductionVertex[iDecay].Branch("etaErr", &etaErr_KFProductionVertex[iDecay], "etaErr[nTracks]/F");
    treeKFProductionVertex[iDecay].Branch("phiErr", &phiErr_KFProductionVertex[iDecay], "phiErr[nTracks]/F");
    treeKFProductionVertex[iDecay].Branch("massErr", &massErr_KFProductionVertex[iDecay], "massErr[nTracks]/F");
    treeKFProductionVertex[iDecay].Branch("decayLengthErr", &decayLengthErr_KFProductionVertex[iDecay],
                                          "decayLengthErr[nTracks]/F");
    treeKFProductionVertex[iDecay].Branch("decayLengthXYErr", &decayLengthXYErr_KFProductionVertex[iDecay],
                                          "decayLengthXYErr[nTracks]/F");
    treeKFProductionVertex[iDecay].Branch("lifeTimeErr", &lifeTimeErr_KFProductionVertex[iDecay],
                                          "lifeTimeErr[nTracks]/F");
    treeKFProductionVertex[iDecay].Branch("rErr", &rErr_KFProductionVertex[iDecay], "rErr[nTracks]/F");
  }

  TTree treeKFDecayVertex[NUM_OF_DECAYS];  // = new TTree("KFBeforeReco", "KFBeforeReco");
  for (int iDecay = 0; iDecay < NUM_OF_DECAYS; iDecay++) {
    treeKFDecayVertex[iDecay].SetObject(nameOfKFDecayVertexTree[iDecay], nameOfKFDecayVertexTree[iDecay]);
  }

  int nTracks_KFDecayVertex[NUM_OF_DECAYS];
  int numOfMCEvent_KFDecayVertex[NUM_OF_DECAYS];
  bool isAllCovMatOK_KFDecayVertex[NUM_OF_DECAYS];

  int id_KFDecayVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  int idMC_KFDecayVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  int pdg_KFDecayVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float params_KFDecayVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS][6];
  float covMat_KFDecayVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS][36];
  bool isCovMatOK_KFDecayVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float x_KFDecayVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float y_KFDecayVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float z_KFDecayVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float pX_KFDecayVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float pY_KFDecayVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float pZ_KFDecayVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float e_KFDecayVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float s_KFDecayVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  int charge_KFDecayVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float chi2_KFDecayVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float ndf_KFDecayVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float massHypo_KFDecayVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];

  bool isMother_KFDecayVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  bool isDaughter_KFDecayVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];

  float p_KFDecayVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float pT_KFDecayVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float eta_KFDecayVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float phi_KFDecayVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float mass_KFDecayVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float decayLength_KFDecayVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float decayLengthXY_KFDecayVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float lifeTime_KFDecayVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float r_KFDecayVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float rapidity_KFDecayVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float theta_KFDecayVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];

  float xErr_KFDecayVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float yErr_KFDecayVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float zErr_KFDecayVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float pXErr_KFDecayVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float pYErr_KFDecayVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float pZErr_KFDecayVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float eErr_KFDecayVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float sErr_KFDecayVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float pErr_KFDecayVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float pTErr_KFDecayVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float etaErr_KFDecayVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float phiErr_KFDecayVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float massErr_KFDecayVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float decayLengthErr_KFDecayVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float decayLengthXYErr_KFDecayVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float lifeTimeErr_KFDecayVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];
  float rErr_KFDecayVertex[NUM_OF_DECAYS][1 + MAX_NUM_OF_DAUGHTERS];

  for (int iDecay = 0; iDecay < NUM_OF_DECAYS; iDecay++) {
    treeKFDecayVertex[iDecay].Branch("nTracks", &nTracks_KFDecayVertex[iDecay], "nTracks/I");
    treeKFDecayVertex[iDecay].Branch("numOfMCEvent", &numOfMCEvent_KFDecayVertex[iDecay], "numOfMCEvent/I");
    treeKFDecayVertex[iDecay].Branch("isAllCovMatOK", &isAllCovMatOK_KFDecayVertex[iDecay], "isAllCovMatOK/O");

    treeKFDecayVertex[iDecay].Branch("ID", id_KFDecayVertex[iDecay], "ID[nTracks]/I");
    treeKFDecayVertex[iDecay].Branch("IDMC", idMC_KFDecayVertex[iDecay], "IDMC[nTracks]/I");
    treeKFDecayVertex[iDecay].Branch("pdgID", pdg_KFDecayVertex[iDecay], "pdgID[nTracks]/I");
    treeKFDecayVertex[iDecay].Branch("params", &params_KFDecayVertex[iDecay], "params[nTracks][6]/F");
    treeKFDecayVertex[iDecay].Branch("covMat", &covMat_KFDecayVertex[iDecay], "covMat[nTracks][36]/F");
    treeKFDecayVertex[iDecay].Branch("isCovMatOK", &isCovMatOK_KFDecayVertex[iDecay], "isCovMatOK[nTracks]/O");
    treeKFDecayVertex[iDecay].Branch("x", &x_KFDecayVertex[iDecay], "x[nTracks]/F");
    treeKFDecayVertex[iDecay].Branch("y", &y_KFDecayVertex[iDecay], "y[nTracks]/F");
    treeKFDecayVertex[iDecay].Branch("z", &z_KFDecayVertex[iDecay], "z[nTracks]/F");
    treeKFDecayVertex[iDecay].Branch("pX", &pX_KFDecayVertex[iDecay], "pX[nTracks]/F");
    treeKFDecayVertex[iDecay].Branch("pY", &pY_KFDecayVertex[iDecay], "pY[nTracks]/F");
    treeKFDecayVertex[iDecay].Branch("pZ", &pZ_KFDecayVertex[iDecay], "pZ[nTracks]/F");
    treeKFDecayVertex[iDecay].Branch("e", &e_KFDecayVertex[iDecay], "e[nTracks]/F");
    treeKFDecayVertex[iDecay].Branch("s", &s_KFDecayVertex[iDecay], "s[nTracks]/F");
    treeKFDecayVertex[iDecay].Branch("charge", &charge_KFDecayVertex[iDecay], "charge[nTracks]/I");
    treeKFDecayVertex[iDecay].Branch("chi2", &chi2_KFDecayVertex[iDecay], "chi2[nTracks]/F");
    treeKFDecayVertex[iDecay].Branch("ndf", &ndf_KFDecayVertex[iDecay], "ndf[nTracks]/F");
    treeKFDecayVertex[iDecay].Branch("massHypo", &massHypo_KFDecayVertex[iDecay], "massHypo[nTracks]/F");

    treeKFDecayVertex[iDecay].Branch("isMother", &isMother_KFDecayVertex[iDecay], "isMother[nTracks]/O");
    treeKFDecayVertex[iDecay].Branch("isDaughter", &isDaughter_KFDecayVertex[iDecay], "isDaughter[nTracks]/O");

    treeKFDecayVertex[iDecay].Branch("p", &p_KFDecayVertex[iDecay], "p[nTracks]/F");
    treeKFDecayVertex[iDecay].Branch("pT", &pT_KFDecayVertex[iDecay], "pT[nTracks]/F");
    treeKFDecayVertex[iDecay].Branch("eta", &eta_KFDecayVertex[iDecay], "eta[nTracks]/F");
    treeKFDecayVertex[iDecay].Branch("phi", &phi_KFDecayVertex[iDecay], "phi[nTracks]/F");
    treeKFDecayVertex[iDecay].Branch("mass", &mass_KFDecayVertex[iDecay], "mass[nTracks]/F");
    treeKFDecayVertex[iDecay].Branch("decayLength", &decayLength_KFDecayVertex[iDecay], "decayLength[nTracks]/F");
    treeKFDecayVertex[iDecay].Branch("decayLengthXY", &decayLengthXY_KFDecayVertex[iDecay], "decayLengthXY[nTracks]/F");
    treeKFDecayVertex[iDecay].Branch("lifeTime", &lifeTime_KFDecayVertex[iDecay], "lifeTime[nTracks]/F");
    treeKFDecayVertex[iDecay].Branch("r", &r_KFDecayVertex[iDecay], "r[nTracks]/F");
    treeKFDecayVertex[iDecay].Branch("rapidity", &rapidity_KFDecayVertex[iDecay], "rapidity[nTracks]/F");
    treeKFDecayVertex[iDecay].Branch("theta", &theta_KFDecayVertex[iDecay], "theta[nTracks]/F");

    treeKFDecayVertex[iDecay].Branch("xErr", &xErr_KFDecayVertex[iDecay], "xErr[nTracks]/F");
    treeKFDecayVertex[iDecay].Branch("yErr", &yErr_KFDecayVertex[iDecay], "yErr[nTracks]/F");
    treeKFDecayVertex[iDecay].Branch("zErr", &zErr_KFDecayVertex[iDecay], "zErr[nTracks]/F");
    treeKFDecayVertex[iDecay].Branch("pXErr", &pXErr_KFDecayVertex[iDecay], "pXErr[nTracks]/F");
    treeKFDecayVertex[iDecay].Branch("pYErr", &pYErr_KFDecayVertex[iDecay], "pYErr[nTracks]/F");
    treeKFDecayVertex[iDecay].Branch("pZErr", &pZErr_KFDecayVertex[iDecay], "pZErr[nTracks]/F");
    treeKFDecayVertex[iDecay].Branch("eErr", &eErr_KFDecayVertex[iDecay], "eErr[nTracks]/F");
    treeKFDecayVertex[iDecay].Branch("sErr", &sErr_KFDecayVertex[iDecay], "sErr[nTracks]/F");
    treeKFDecayVertex[iDecay].Branch("pErr", &pErr_KFDecayVertex[iDecay], "pErr[nTracks]/F");
    treeKFDecayVertex[iDecay].Branch("pTErr", &pTErr_KFDecayVertex[iDecay], "pTErr[nTracks]/F");
    treeKFDecayVertex[iDecay].Branch("etaErr", &etaErr_KFDecayVertex[iDecay], "etaErr[nTracks]/F");
    treeKFDecayVertex[iDecay].Branch("phiErr", &phiErr_KFDecayVertex[iDecay], "phiErr[nTracks]/F");
    treeKFDecayVertex[iDecay].Branch("massErr", &massErr_KFDecayVertex[iDecay], "massErr[nTracks]/F");
    treeKFDecayVertex[iDecay].Branch("decayLengthErr", &decayLengthErr_KFDecayVertex[iDecay],
                                     "decayLengthErr[nTracks]/F");
    treeKFDecayVertex[iDecay].Branch("decayLengthXYErr", &decayLengthXYErr_KFDecayVertex[iDecay],
                                     "decayLengthXYErr[nTracks]/F");
    treeKFDecayVertex[iDecay].Branch("lifeTimeErr", &lifeTimeErr_KFDecayVertex[iDecay], "lifeTimeErr[nTracks]/F");
    treeKFDecayVertex[iDecay].Branch("rErr", &rErr_KFDecayVertex[iDecay], "rErr[nTracks]/F");
  }


  int nEvents = chain.GetEntries();

  cout << endl;
  cout << "Starting the event loop\n";
  ////////////////////////////////////////////////////////
  //          START OF THE EVENT LOOP
  ////////////////////////////////////////////////////////
  CovMatrixCheckStats stats;

  for (int iEvent = 0; iEvent < nEvents; iEvent++) {
    chain.GetEntry(iEvent);

    // print a processing progress
    if (iEvent % 1000 == 0) {
      std::cout << "processing " << iEvent << " event\r";
      std::cout << std::flush;
    }


    // create decay keys
    std::array<int, NUM_OF_DECAYS> decay_mother_id;

    for (int iTrack = 0; iTrack < nTracks_MC; iTrack++) {
      for (int iDecay = 0; iDecay < NUM_OF_DECAYS; iDecay++) {
        if (pdg_MC[iTrack] == MOTHER_PDG_IDS[iDecay]) { decay_mother_id[iDecay] = trackID_MC[iTrack]; }
      }
    }

    // create MC particles structure
    std::array<MCParticleStruct, NUM_OF_DECAYS> mother_MC;
    std::array<std::array<MCParticleStruct, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> daughters_MC;

    // create KFParticle structure for particles
    std::array<KFParticle, NUM_OF_DECAYS> mother_KF;
    std::array<std::array<KFParticle, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> daughters_KF;

    std::array<int, NUM_OF_DECAYS> daughter_counter;
    for (int iDecay = 0; iDecay < NUM_OF_DECAYS; iDecay++) { daughter_counter[iDecay] = 0; }

    // loop over the tracks
    for (int iTrack = 0; iTrack < nTracks_MC; iTrack++) {
      for (int iDecay = 0; iDecay < NUM_OF_DECAYS; iDecay++) {
        if (trackID_MC[iTrack] == decay_mother_id[iDecay]) {
          //this is a mother track in decay number iDecay
          mother_MC[iDecay].trackID  = iTrack;  //trackID_MC[iTrack];
          mother_MC[iDecay].parentID = parentID_MC[iTrack];
          mother_MC[iDecay].pdg      = pdg_MC[iTrack];
          //mother_MC[iDecay].pdgLifeTime = pdgLifeTime_MC[iTrack];
          mother_MC[iDecay].charge      = charge_MC[iTrack];
          mother_MC[iDecay].isMother    = true;
          mother_MC[iDecay].initialX    = initialX_MC[iTrack];
          mother_MC[iDecay].initialY    = initialY_MC[iTrack];
          mother_MC[iDecay].initialZ    = initialZ_MC[iTrack];
          mother_MC[iDecay].finalX      = finalX_MC[iTrack];
          mother_MC[iDecay].finalY      = finalY_MC[iTrack];
          mother_MC[iDecay].finalZ      = finalZ_MC[iTrack];
          mother_MC[iDecay].trackLength = trackLength_MC[iTrack];
          mother_MC[iDecay].initialPX   = initialPX_MC[iTrack];
          mother_MC[iDecay].initialPY   = initialPY_MC[iTrack];
          mother_MC[iDecay].initialPZ   = initialPZ_MC[iTrack];
          mother_MC[iDecay].finalPX     = finalPX_MC[iTrack];
          mother_MC[iDecay].finalPY     = finalPY_MC[iTrack];
          mother_MC[iDecay].finalPZ     = finalPZ_MC[iTrack];
          mother_MC[iDecay].mass        = mass_MC[iTrack];

          // fill MC Initial tree
          id_MCInitial[iDecay][0]            = trackID_MC[iTrack];
          parentid_MCInitial[iDecay][0]      = parentID_MC[iTrack];
          pdg_MCInitial[iDecay][0]           = pdg_MC[iTrack];
          x_MCInitial[iDecay][0]             = initialX_MC[iTrack];
          y_MCInitial[iDecay][0]             = initialY_MC[iTrack];
          z_MCInitial[iDecay][0]             = initialZ_MC[iTrack];
          pX_MCInitial[iDecay][0]            = initialPX_MC[iTrack];
          pY_MCInitial[iDecay][0]            = initialPY_MC[iTrack];
          pZ_MCInitial[iDecay][0]            = initialPZ_MC[iTrack];
          e_MCInitial[iDecay][0]             = 0;  //TODO
          s_MCInitial[iDecay][0]             = 0;  //TODO
          charge_MCInitial[iDecay][0]        = charge_MC[iTrack];
          isMother_MCInitial[iDecay][0]      = true;
          isDaughter_MCInitial[iDecay][0]    = false;
          p_MCInitial[iDecay][0]             = 0;  //TODO
          pT_MCInitial[iDecay][0]            = 0;  //TODO
          eta_MCInitial[iDecay][0]           = 0;  //TODO
          phi_MCInitial[iDecay][0]           = 0;  //TODO
          mass_MCInitial[iDecay][0]          = mass_MC[iTrack];
          decayLength_MCInitial[iDecay][0]   = trackLength_MC[iTrack];  //TODO
          decayLengthXY_MCInitial[iDecay][0] = 0;                       //TODO
          lifeTime_MCInitial[iDecay][0]      = 0;                       //TODO
          r_MCInitial[iDecay][0]             = 0;                       //TODO
          rapidity_MCInitial[iDecay][0]      = 0;                       //TODO
          theta_MCInitial[iDecay][0]         = 0;                       //TODO

          // fill MC Final tree
          id_MCFinal[iDecay][0]            = trackID_MC[iTrack];
          parentid_MCFinal[iDecay][0]      = parentID_MC[iTrack];
          pdg_MCFinal[iDecay][0]           = pdg_MC[iTrack];
          x_MCFinal[iDecay][0]             = finalX_MC[iTrack];
          y_MCFinal[iDecay][0]             = finalY_MC[iTrack];
          z_MCFinal[iDecay][0]             = finalZ_MC[iTrack];
          pX_MCFinal[iDecay][0]            = finalPX_MC[iTrack];
          pY_MCFinal[iDecay][0]            = finalPY_MC[iTrack];
          pZ_MCFinal[iDecay][0]            = finalPZ_MC[iTrack];
          e_MCFinal[iDecay][0]             = 0;  //TODO
          s_MCFinal[iDecay][0]             = 0;  //TODO
          charge_MCFinal[iDecay][0]        = charge_MC[iTrack];
          isMother_MCFinal[iDecay][0]      = true;
          isDaughter_MCFinal[iDecay][0]    = false;
          p_MCFinal[iDecay][0]             = 0;  //TODO
          pT_MCFinal[iDecay][0]            = 0;  //TODO
          eta_MCFinal[iDecay][0]           = 0;  //TODO
          phi_MCFinal[iDecay][0]           = 0;  //TODO
          mass_MCFinal[iDecay][0]          = mass_MC[iTrack];
          decayLength_MCFinal[iDecay][0]   = trackLength_MC[iTrack];  //TODO
          decayLengthXY_MCFinal[iDecay][0] = 0;                       //TODO
          lifeTime_MCFinal[iDecay][0]      = 0;                       //TODO
          r_MCFinal[iDecay][0]             = 0;                       //TODO
          rapidity_MCFinal[iDecay][0]      = 0;                       //TODO
          theta_MCFinal[iDecay][0]         = 0;                       //TODO
        }
        if (parentID_MC[iTrack] == decay_mother_id[iDecay]) {
          //this is a daughter track in decay number iDecay
          daughters_MC[iDecay][daughter_counter[iDecay]].trackID  = iTrack;  //trackID_MC[iTrack];
          daughters_MC[iDecay][daughter_counter[iDecay]].parentID = parentID_MC[iTrack];
          daughters_MC[iDecay][daughter_counter[iDecay]].pdg      = pdg_MC[iTrack];
          //daughters_MC[iDecay][daughter_counter[iDecay]].pdgLifeTime = pdgLifeTime_MC[iTrack];
          daughters_MC[iDecay][daughter_counter[iDecay]].charge      = charge_MC[iTrack];
          daughters_MC[iDecay][daughter_counter[iDecay]].isMother    = false;
          daughters_MC[iDecay][daughter_counter[iDecay]].initialX    = initialX_MC[iTrack];
          daughters_MC[iDecay][daughter_counter[iDecay]].initialY    = initialY_MC[iTrack];
          daughters_MC[iDecay][daughter_counter[iDecay]].initialZ    = initialZ_MC[iTrack];
          daughters_MC[iDecay][daughter_counter[iDecay]].finalX      = finalX_MC[iTrack];
          daughters_MC[iDecay][daughter_counter[iDecay]].finalY      = finalY_MC[iTrack];
          daughters_MC[iDecay][daughter_counter[iDecay]].finalZ      = finalZ_MC[iTrack];
          daughters_MC[iDecay][daughter_counter[iDecay]].trackLength = trackLength_MC[iTrack];
          daughters_MC[iDecay][daughter_counter[iDecay]].initialPX   = initialPX_MC[iTrack];
          daughters_MC[iDecay][daughter_counter[iDecay]].initialPY   = initialPY_MC[iTrack];
          daughters_MC[iDecay][daughter_counter[iDecay]].initialPZ   = initialPZ_MC[iTrack];
          daughters_MC[iDecay][daughter_counter[iDecay]].finalPX     = finalPX_MC[iTrack];
          daughters_MC[iDecay][daughter_counter[iDecay]].finalPY     = finalPY_MC[iTrack];
          daughters_MC[iDecay][daughter_counter[iDecay]].finalPZ     = finalPZ_MC[iTrack];
          daughters_MC[iDecay][daughter_counter[iDecay]].mass        = mass_MC[iTrack];
          daughter_counter[iDecay]++;

          // fill MC Initial tree
          id_MCInitial[iDecay][daughter_counter[iDecay]]            = trackID_MC[iTrack];
          parentid_MCInitial[iDecay][daughter_counter[iDecay]]      = parentID_MC[iTrack];
          pdg_MCInitial[iDecay][daughter_counter[iDecay]]           = pdg_MC[iTrack];
          x_MCInitial[iDecay][daughter_counter[iDecay]]             = initialX_MC[iTrack];
          y_MCInitial[iDecay][daughter_counter[iDecay]]             = initialY_MC[iTrack];
          z_MCInitial[iDecay][daughter_counter[iDecay]]             = initialZ_MC[iTrack];
          pX_MCInitial[iDecay][daughter_counter[iDecay]]            = initialPX_MC[iTrack];
          pY_MCInitial[iDecay][daughter_counter[iDecay]]            = initialPY_MC[iTrack];
          pZ_MCInitial[iDecay][daughter_counter[iDecay]]            = initialPZ_MC[iTrack];
          e_MCInitial[iDecay][daughter_counter[iDecay]]             = 0;  //TODO
          s_MCInitial[iDecay][daughter_counter[iDecay]]             = 0;  //TODO
          charge_MCInitial[iDecay][daughter_counter[iDecay]]        = charge_MC[iTrack];
          isMother_MCInitial[iDecay][daughter_counter[iDecay]]      = false;
          isDaughter_MCInitial[iDecay][daughter_counter[iDecay]]    = true;
          p_MCInitial[iDecay][daughter_counter[iDecay]]             = 0;  //TODO
          pT_MCInitial[iDecay][daughter_counter[iDecay]]            = 0;  //TODO
          eta_MCInitial[iDecay][daughter_counter[iDecay]]           = 0;  //TODO
          phi_MCInitial[iDecay][daughter_counter[iDecay]]           = 0;  //TODO
          mass_MCInitial[iDecay][daughter_counter[iDecay]]          = mass_MC[iTrack];
          decayLength_MCInitial[iDecay][daughter_counter[iDecay]]   = 0;  //TODO
          decayLengthXY_MCInitial[iDecay][daughter_counter[iDecay]] = 0;  //TODO
          lifeTime_MCInitial[iDecay][daughter_counter[iDecay]]      = 0;  //TODO
          r_MCInitial[iDecay][daughter_counter[iDecay]]             = 0;  //TODO
          rapidity_MCInitial[iDecay][daughter_counter[iDecay]]      = 0;  //TODO
          theta_MCInitial[iDecay][daughter_counter[iDecay]]         = 0;  //TODO

          // fill MC Final tree
          id_MCFinal[iDecay][daughter_counter[iDecay]]            = trackID_MC[iTrack];
          parentid_MCFinal[iDecay][daughter_counter[iDecay]]      = parentID_MC[iTrack];
          pdg_MCFinal[iDecay][daughter_counter[iDecay]]           = pdg_MC[iTrack];
          x_MCFinal[iDecay][daughter_counter[iDecay]]             = finalX_MC[iTrack];
          y_MCFinal[iDecay][daughter_counter[iDecay]]             = finalY_MC[iTrack];
          z_MCFinal[iDecay][daughter_counter[iDecay]]             = finalZ_MC[iTrack];
          pX_MCFinal[iDecay][daughter_counter[iDecay]]            = finalPX_MC[iTrack];
          pY_MCFinal[iDecay][daughter_counter[iDecay]]            = finalPY_MC[iTrack];
          pZ_MCFinal[iDecay][daughter_counter[iDecay]]            = finalPZ_MC[iTrack];
          e_MCFinal[iDecay][daughter_counter[iDecay]]             = 0;  //TODO
          s_MCFinal[iDecay][daughter_counter[iDecay]]             = 0;  //TODO
          charge_MCFinal[iDecay][daughter_counter[iDecay]]        = charge_MC[iTrack];
          isMother_MCFinal[iDecay][daughter_counter[iDecay]]      = false;
          isDaughter_MCFinal[iDecay][daughter_counter[iDecay]]    = true;
          p_MCFinal[iDecay][daughter_counter[iDecay]]             = 0;  //TODO
          pT_MCFinal[iDecay][daughter_counter[iDecay]]            = 0;  //TODO
          eta_MCFinal[iDecay][daughter_counter[iDecay]]           = 0;  //TODO
          phi_MCFinal[iDecay][daughter_counter[iDecay]]           = 0;  //TODO
          mass_MCFinal[iDecay][daughter_counter[iDecay]]          = mass_MC[iTrack];
          decayLength_MCFinal[iDecay][daughter_counter[iDecay]]   = 0;  //TODO
          decayLengthXY_MCFinal[iDecay][daughter_counter[iDecay]] = 0;  //TODO
          lifeTime_MCFinal[iDecay][daughter_counter[iDecay]]      = 0;  //TODO
          r_MCFinal[iDecay][daughter_counter[iDecay]]             = 0;  //TODO
          rapidity_MCFinal[iDecay][daughter_counter[iDecay]]      = 0;  //TODO
          theta_MCFinal[iDecay][daughter_counter[iDecay]]         = 0;  //TODO
        }
      }
    }  // end of track loop

    for (int iDecay = 0; iDecay < NUM_OF_DECAYS; iDecay++) {
      nTracks_MCInitial[iDecay]      = 1 + daughter_counter[iDecay];
      nTracks_MCFinal[iDecay]        = 1 + daughter_counter[iDecay];
      numOfMCEvent_MCInitial[iDecay] = iEvent;
      numOfMCEvent_MCFinal[iDecay]   = iEvent;
      treeMCInitial[iDecay].Fill();
      treeMCFinal[iDecay].Fill();
    }

    float testCovMat[36];
    bool testmode = true;
    bool is_silent = true;

    for (int iDecay = 0; iDecay < NUM_OF_DECAYS; iDecay++) {
      /// Create KFParticles from MCParticleStructs
      float params_M[6] = {mother_MC[iDecay].finalX,  mother_MC[iDecay].finalY,  mother_MC[iDecay].finalZ,
                           mother_MC[iDecay].finalPX, mother_MC[iDecay].finalPY, mother_MC[iDecay].finalPZ};
      std::vector<float> covmat_vec_ = MakeCovMatrix(mother_MC[iDecay]);
      float covmat_M[21];
      std::copy(covmat_vec_.begin(), covmat_vec_.end(), covmat_M);
      int charge_ = mother_MC[iDecay].charge;
      float chi2_ = mother_MC[iDecay].chi2;
      int ndf_    = mother_MC[iDecay].ndf;
      float mass_ = mother_MC[iDecay].mass;
      mother_KF[iDecay].SetId(mother_MC[iDecay].trackID);
      mother_KF[iDecay].SetPDG(mother_MC[iDecay].pdg);

      if (testmode) {
        for (int idx = 0; idx < 36; idx++) { testCovMat[idx] = mother_KF[iDecay].GetCovariance(idx); }
        CheckCovMat(testCovMat, stats, is_silent);
      }

      for (int i = 0; i < daughter_counter[iDecay]; i++) {
        float params_D[6] = {daughters_MC[iDecay][i].initialX,  daughters_MC[iDecay][i].initialY,
                             daughters_MC[iDecay][i].initialZ,  daughters_MC[iDecay][i].initialPX,
                             daughters_MC[iDecay][i].initialPY, daughters_MC[iDecay][i].initialPZ};
        covmat_vec_       = MakeCovMatrix(daughters_MC[iDecay][i]);
        float covmat_D[21];
        std::copy(covmat_vec_.begin(), covmat_vec_.end(), covmat_D);
        charge_ = daughters_MC[iDecay][i].charge;
        chi2_   = 0;  //daughters_MC[iDecay][i].chi2;
        ndf_    = 0;  //daughters_MC[iDecay][i].ndf;
        mass_   = daughters_MC[iDecay][i].mass;
#ifdef TESTSUITE
        daughters_KF[iDecay][i].Initialize(params_D, covmat_D, charge_, chi2_, ndf_, mass_);
#else
        daughters_KF[iDecay][i].Create(params_D, covmat_D, charge_, mass_);
#endif
        daughters_KF[iDecay][i].SetId(daughters_MC[iDecay][i].trackID);
        daughters_KF[iDecay][i].SetPDG(daughters_MC[iDecay][i].pdg);

        if (testmode) {
          for (int idx = 0; idx < 36; idx++) { testCovMat[idx] = daughters_KF[iDecay][i].GetCovariance(idx); }
          CheckCovMat(testCovMat, stats, is_silent);
        }
      }

      // do particle reconstruction here
      mother_KF[iDecay].SetConstructMethod(CONSTRUCT_METHOD_NUMBER);

      bool isDecomposedOK = true;
      for (auto& part : daughters_KF[iDecay]) {
        isDecomposedOK = SmearParticle(part);
        mother_KF[iDecay].AddDaughter(part);
      }

      if (testmode) {
        for (int idx = 0; idx < 36; idx++) { testCovMat[idx] = mother_KF[iDecay].GetCovariance(idx); }
        CheckCovMat(testCovMat, stats, is_silent);
      }

      if (SET_TOPOLOGICAL_CONSTRAINT_MOTHER) {
        KFPVertex vert;
        vert.SetXYZ(mother_MC[iDecay].initialX, mother_MC[iDecay].initialY, mother_MC[iDecay].initialZ);
        vert.SetCovarianceMatrix(0.1, 0., 0.1, 0., 0., 0.1);  // ????
        // vert.SetCovarianceMatrix(
        //         mother_KF[iDecay].GetCovariance(0),
        //         mother_KF[iDecay].GetCovariance(1),
        //         mother_KF[iDecay].GetCovariance(2),
        //         mother_KF[iDecay].GetCovariance(3),
        //         mother_KF[iDecay].GetCovariance(4),
        //         mother_KF[iDecay].GetCovariance(5)
        //     ); // ????

        // #ifdef MAKE_VERTEX_COV_MAT
        std::vector<float> covmat_vec_vertex_ = MakeVertexCovMatrix(mother_MC[iDecay]);
        float covmat_vec_vertex_array_[6];

        //std::copy(covmat_vec_vertex_.begin(), covmat_vec_vertex_.end(), covmat_vec_vertex_array_);
        //vert.SetCovarianceMatrix(covmat_vec_vertex_array_);
        SmearVertex(vert);
        //  #endif
        mother_KF[iDecay].SetProductionVertex(KFVertex(vert));

        if (testmode) {
          for (int idx = 0; idx < 36; idx++) { testCovMat[idx] = mother_KF[iDecay].GetCovariance(idx); }
          CheckCovMat(testCovMat, stats, is_silent);
        }
      }

      //mother_KF[iDecay].TransportToProductionVertex();
      float point[3] = {mother_MC[iDecay].initialX, mother_MC[iDecay].initialY, mother_MC[iDecay].initialZ};
      mother_KF[iDecay].TransportToPoint(point);

      if (testmode) {
        for (int idx = 0; idx < 36; idx++) { testCovMat[idx] = mother_KF[iDecay].GetCovariance(idx); }
        CheckCovMat(testCovMat, stats, is_silent);
      }

      // Fill KFProductionVertex tree
      nTracks_KFProductionVertex[iDecay]       = 1 + daughter_counter[iDecay];
      numOfMCEvent_KFProductionVertex[iDecay]  = iEvent;
      isAllCovMatOK_KFProductionVertex[iDecay] = true;

      id_KFProductionVertex[iDecay][0]   = mother_KF[iDecay].Id();
      idMC_KFProductionVertex[iDecay][0] = mother_MC[iDecay].trackID;
      pdg_KFProductionVertex[iDecay][0]  = mother_KF[iDecay].GetPDG();
      // for (int iPar = 0; iPar < 6; iPar++)
      //     params_KFProductionVertex[iDecay][0][iPar] = mother_KF[iDecay].GetParameter(iPar);
      // for (int iCov = 0; iCov < 36; iCov++)
      //     covMat_KFProductionVertex[iDecay][0][iCov] = mother_KF[iDecay].GetCovariance(iCov);
      // isCovMatOK_KFProductionVertex[iDecay][0] = CheckCovMat(covMat_KFProductionVertex[iDecay][0], stats);
      //  if (!isCovMatOK_KFProductionVertex[iDecay][0])
      //   isAllCovMatOK_KFProductionVertex[iDecay] = false;

      x_KFProductionVertex[iDecay][0]        = mother_KF[iDecay].GetX();
      y_KFProductionVertex[iDecay][0]        = mother_KF[iDecay].GetY();
      z_KFProductionVertex[iDecay][0]        = mother_KF[iDecay].GetZ();
      pX_KFProductionVertex[iDecay][0]       = mother_KF[iDecay].GetPx();
      pY_KFProductionVertex[iDecay][0]       = mother_KF[iDecay].GetPy();
      pZ_KFProductionVertex[iDecay][0]       = mother_KF[iDecay].GetPz();
      e_KFProductionVertex[iDecay][0]        = mother_KF[iDecay].GetE();
      s_KFProductionVertex[iDecay][0]        = mother_KF[iDecay].GetS();
      charge_KFProductionVertex[iDecay][0]   = (int) mother_KF[iDecay].GetQ();
      chi2_KFProductionVertex[iDecay][0]     = mother_KF[iDecay].GetChi2();
      ndf_KFProductionVertex[iDecay][0]      = mother_KF[iDecay].GetNDF();
      massHypo_KFProductionVertex[iDecay][0] = mother_KF[iDecay].GetMassHypo();

      isMother_KFProductionVertex[iDecay][0]   = true;
      isDaughter_KFProductionVertex[iDecay][0] = false;

      mother_KF[iDecay].GetP(p_KFProductionVertex[iDecay][0], pErr_KFProductionVertex[iDecay][0]);
      mother_KF[iDecay].GetPt(pT_KFProductionVertex[iDecay][0], pTErr_KFProductionVertex[iDecay][0]);
      mother_KF[iDecay].GetEta(eta_KFProductionVertex[iDecay][0], etaErr_KFProductionVertex[iDecay][0]);
      mother_KF[iDecay].GetPhi(phi_KFProductionVertex[iDecay][0], phiErr_KFProductionVertex[iDecay][0]);
      mother_KF[iDecay].GetMass(mass_KFProductionVertex[iDecay][0], massErr_KFProductionVertex[iDecay][0]);
      mother_KF[iDecay].GetDecayLength(decayLength_KFProductionVertex[iDecay][0],
                                       decayLengthErr_KFProductionVertex[iDecay][0]);
      mother_KF[iDecay].GetDecayLengthXY(decayLengthXY_KFProductionVertex[iDecay][0],
                                         decayLengthXYErr_KFProductionVertex[iDecay][0]);
      mother_KF[iDecay].GetLifeTime(lifeTime_KFProductionVertex[iDecay][0], lifeTimeErr_KFProductionVertex[iDecay][0]);
      mother_KF[iDecay].GetR(r_KFProductionVertex[iDecay][0], rErr_KFProductionVertex[iDecay][0]);

      rapidity_KFProductionVertex[iDecay][0] = mother_KF[iDecay].GetRapidity();
      theta_KFProductionVertex[iDecay][0]    = mother_KF[iDecay].GetTheta();

      xErr_KFProductionVertex[iDecay][0]  = mother_KF[iDecay].GetErrX();
      yErr_KFProductionVertex[iDecay][0]  = mother_KF[iDecay].GetErrY();
      zErr_KFProductionVertex[iDecay][0]  = mother_KF[iDecay].GetErrZ();
      pXErr_KFProductionVertex[iDecay][0] = mother_KF[iDecay].GetErrPx();
      pYErr_KFProductionVertex[iDecay][0] = mother_KF[iDecay].GetErrPy();
      pZErr_KFProductionVertex[iDecay][0] = mother_KF[iDecay].GetErrPz();
      eErr_KFProductionVertex[iDecay][0]  = mother_KF[iDecay].GetErrE();
      sErr_KFProductionVertex[iDecay][0]  = mother_KF[iDecay].GetErrS();

      for (int i = 0; i < daughter_counter[iDecay]; i++) {
        int j                              = i + 1;
        id_KFProductionVertex[iDecay][j]   = daughters_KF[iDecay][i].Id();
        idMC_KFProductionVertex[iDecay][j] = daughters_MC[iDecay][i].trackID;
        pdg_KFProductionVertex[iDecay][j]  = daughters_KF[iDecay][i].GetPDG();
        for (int iPar = 0; iPar < 6; iPar++) {
          params_KFProductionVertex[iDecay][j][iPar] = daughters_KF[iDecay][i].GetParameter(iPar);
        }
        for (int iCov = 0; iCov < 36; iCov++) {
          covMat_KFProductionVertex[iDecay][j][iCov] = daughters_KF[iDecay][i].GetCovariance(iCov);
        }
        isCovMatOK_KFProductionVertex[iDecay][j] = CheckCovMat(covMat_KFProductionVertex[iDecay][j], stats);
        // if (!isCovMatOK_KFProductionVertex[iDecay][j])
        // isAllCovMatOK_KFProductionVertex[iDecay] = false;
        x_KFProductionVertex[iDecay][j]        = daughters_KF[iDecay][i].GetX();
        y_KFProductionVertex[iDecay][j]        = daughters_KF[iDecay][i].GetY();
        z_KFProductionVertex[iDecay][j]        = daughters_KF[iDecay][i].GetZ();
        pX_KFProductionVertex[iDecay][j]       = daughters_KF[iDecay][i].GetPx();
        pY_KFProductionVertex[iDecay][j]       = daughters_KF[iDecay][i].GetPy();
        pZ_KFProductionVertex[iDecay][j]       = daughters_KF[iDecay][i].GetPz();
        e_KFProductionVertex[iDecay][j]        = daughters_KF[iDecay][i].GetE();
        s_KFProductionVertex[iDecay][j]        = daughters_KF[iDecay][i].GetS();
        charge_KFProductionVertex[iDecay][j]   = (int) daughters_KF[iDecay][i].GetQ();
        chi2_KFProductionVertex[iDecay][j]     = daughters_KF[iDecay][i].GetChi2();
        ndf_KFProductionVertex[iDecay][j]      = daughters_KF[iDecay][i].GetNDF();
        massHypo_KFProductionVertex[iDecay][j] = daughters_KF[iDecay][i].GetMassHypo();

        isMother_KFProductionVertex[iDecay][j]   = false;
        isDaughter_KFProductionVertex[iDecay][j] = true;

        daughters_KF[iDecay][i].GetP(p_KFProductionVertex[iDecay][j], pErr_KFProductionVertex[iDecay][j]);
        daughters_KF[iDecay][i].GetPt(pT_KFProductionVertex[iDecay][j], pTErr_KFProductionVertex[iDecay][j]);
        daughters_KF[iDecay][i].GetEta(eta_KFProductionVertex[iDecay][j], etaErr_KFProductionVertex[iDecay][j]);
        daughters_KF[iDecay][i].GetPhi(phi_KFProductionVertex[iDecay][j], phiErr_KFProductionVertex[iDecay][j]);
        daughters_KF[iDecay][i].GetMass(mass_KFProductionVertex[iDecay][j], massErr_KFProductionVertex[iDecay][j]);
        daughters_KF[iDecay][i].GetDecayLength(decayLength_KFProductionVertex[iDecay][j],
                                               decayLengthErr_KFProductionVertex[iDecay][j]);
        daughters_KF[iDecay][i].GetDecayLengthXY(decayLengthXY_KFProductionVertex[iDecay][j],
                                                 decayLengthXYErr_KFProductionVertex[iDecay][j]);
        daughters_KF[iDecay][i].GetLifeTime(lifeTime_KFProductionVertex[iDecay][j],
                                            lifeTimeErr_KFProductionVertex[iDecay][j]);
        daughters_KF[iDecay][i].GetR(r_KFProductionVertex[iDecay][j], rErr_KFProductionVertex[iDecay][j]);

        rapidity_KFProductionVertex[iDecay][j] = daughters_KF[iDecay][i].GetRapidity();
        theta_KFProductionVertex[iDecay][j]    = daughters_KF[iDecay][i].GetTheta();

        xErr_KFProductionVertex[iDecay][j]  = daughters_KF[iDecay][i].GetErrX();
        yErr_KFProductionVertex[iDecay][j]  = daughters_KF[iDecay][i].GetErrY();
        zErr_KFProductionVertex[iDecay][j]  = daughters_KF[iDecay][i].GetErrZ();
        pXErr_KFProductionVertex[iDecay][j] = daughters_KF[iDecay][i].GetErrPx();
        pYErr_KFProductionVertex[iDecay][j] = daughters_KF[iDecay][i].GetErrPy();
        pZErr_KFProductionVertex[iDecay][j] = daughters_KF[iDecay][i].GetErrPz();
        eErr_KFProductionVertex[iDecay][j]  = daughters_KF[iDecay][i].GetErrE();
        sErr_KFProductionVertex[iDecay][j]  = daughters_KF[iDecay][i].GetErrS();
      }
      //  mother_KF[iDecay].TransportToDecayVertex();
      float pointL[3] = {mother_MC[iDecay].finalX, mother_MC[iDecay].finalY, mother_MC[iDecay].finalZ};
      mother_KF[iDecay].TransportToPoint(pointL);

      if (testmode) {
        for (int idx = 0; idx < 36; idx++) { testCovMat[idx] = mother_KF[iDecay].GetCovariance(idx); }
        CheckCovMat(testCovMat, stats, is_silent);
      }

      //if (SET_MASS_CONSTRAINT_MOTHER)
      //   mother_KF[iDecay].SetNonlinearMassConstraint(mother_MC[iDecay].mass);

      if (testmode) {
        for (int idx = 0; idx < 36; idx++) { testCovMat[idx] = mother_KF[iDecay].GetCovariance(idx); }
        CheckCovMat(testCovMat, stats, is_silent);
      }

      if (SET_TOPOLOGICAL_CONSTRAINT_DAUGHTERS) {
        for (auto& part : daughters_KF[iDecay]) { part.SetProductionVertex(mother_KF[iDecay]); }
      }

      if (SET_MASS_CONSTRAINT_DAUGHTERS) {
        for (auto& part : daughters_KF[iDecay]) { part.SetNonlinearMassConstraint(part.GetMassHypo()); }
      }

      if (testmode) {
        for (int i = 0; i < daughter_counter[iDecay]; i++) {
          for (int idx = 0; idx < 36; idx++) { testCovMat[idx] = daughters_KF[iDecay][i].GetCovariance(idx); }
          CheckCovMat(testCovMat, stats, is_silent);
        }
      }

      // Fill KFDecayVertex tree
      nTracks_KFProductionVertex[iDecay]  = 1 + daughter_counter[iDecay];
      nTracks_KFDecayVertex[iDecay]       = 1 + daughter_counter[iDecay];
      numOfMCEvent_KFDecayVertex[iDecay]  = iEvent;
      isAllCovMatOK_KFDecayVertex[iDecay] = isDecomposedOK;

      id_KFDecayVertex[iDecay][0]   = mother_KF[iDecay].Id();
      idMC_KFDecayVertex[iDecay][0] = mother_MC[iDecay].trackID;
      pdg_KFDecayVertex[iDecay][0]  = mother_KF[iDecay].GetPDG();
      for (int iPar = 0; iPar < 6; iPar++) {
        params_KFDecayVertex[iDecay][0][iPar] = mother_KF[iDecay].GetParameter(iPar);
      }
      for (int iCov = 0; iCov < 36; iCov++) {
        covMat_KFDecayVertex[iDecay][0][iCov] = mother_KF[iDecay].GetCovariance(iCov);
      }
      isCovMatOK_KFDecayVertex[iDecay][0] = CheckCovMat(covMat_KFDecayVertex[iDecay][0], stats);
      if (!isCovMatOK_KFDecayVertex[iDecay][0]) { isAllCovMatOK_KFDecayVertex[iDecay] = false; }
      x_KFDecayVertex[iDecay][0]        = mother_KF[iDecay].GetX();
      y_KFDecayVertex[iDecay][0]        = mother_KF[iDecay].GetY();
      z_KFDecayVertex[iDecay][0]        = mother_KF[iDecay].GetZ();
      pX_KFDecayVertex[iDecay][0]       = mother_KF[iDecay].GetPx();
      pY_KFDecayVertex[iDecay][0]       = mother_KF[iDecay].GetPy();
      pZ_KFDecayVertex[iDecay][0]       = mother_KF[iDecay].GetPz();
      e_KFDecayVertex[iDecay][0]        = mother_KF[iDecay].GetE();
      s_KFDecayVertex[iDecay][0]        = mother_KF[iDecay].GetS();
      charge_KFDecayVertex[iDecay][0]   = (int) mother_KF[iDecay].GetQ();
      chi2_KFDecayVertex[iDecay][0]     = mother_KF[iDecay].GetChi2();
      ndf_KFDecayVertex[iDecay][0]      = mother_KF[iDecay].GetNDF();
      massHypo_KFDecayVertex[iDecay][0] = mother_KF[iDecay].GetMassHypo();

      isMother_KFDecayVertex[iDecay][0]   = true;
      isDaughter_KFDecayVertex[iDecay][0] = false;

      mother_KF[iDecay].GetP(p_KFDecayVertex[iDecay][0], pErr_KFDecayVertex[iDecay][0]);
      mother_KF[iDecay].GetPt(pT_KFDecayVertex[iDecay][0], pTErr_KFDecayVertex[iDecay][0]);
      mother_KF[iDecay].GetEta(eta_KFDecayVertex[iDecay][0], etaErr_KFDecayVertex[iDecay][0]);
      mother_KF[iDecay].GetPhi(phi_KFDecayVertex[iDecay][0], phiErr_KFDecayVertex[iDecay][0]);
      mother_KF[iDecay].GetMass(mass_KFDecayVertex[iDecay][0], massErr_KFDecayVertex[iDecay][0]);
      mother_KF[iDecay].GetDecayLength(decayLength_KFDecayVertex[iDecay][0], decayLengthErr_KFDecayVertex[iDecay][0]);
      mother_KF[iDecay].GetDecayLengthXY(decayLengthXY_KFDecayVertex[iDecay][0],
                                         decayLengthXYErr_KFDecayVertex[iDecay][0]);
      mother_KF[iDecay].GetLifeTime(lifeTime_KFDecayVertex[iDecay][0], lifeTimeErr_KFDecayVertex[iDecay][0]);
      mother_KF[iDecay].GetR(r_KFDecayVertex[iDecay][0], rErr_KFDecayVertex[iDecay][0]);

      rapidity_KFDecayVertex[iDecay][0] = mother_KF[iDecay].GetRapidity();
      theta_KFDecayVertex[iDecay][0]    = mother_KF[iDecay].GetTheta();

      xErr_KFDecayVertex[iDecay][0]  = mother_KF[iDecay].GetErrX();
      yErr_KFDecayVertex[iDecay][0]  = mother_KF[iDecay].GetErrY();
      zErr_KFDecayVertex[iDecay][0]  = mother_KF[iDecay].GetErrZ();
      pXErr_KFDecayVertex[iDecay][0] = mother_KF[iDecay].GetErrPx();
      pYErr_KFDecayVertex[iDecay][0] = mother_KF[iDecay].GetErrPy();
      pZErr_KFDecayVertex[iDecay][0] = mother_KF[iDecay].GetErrPz();
      eErr_KFDecayVertex[iDecay][0]  = mother_KF[iDecay].GetErrE();
      sErr_KFDecayVertex[iDecay][0]  = mother_KF[iDecay].GetErrS();

      for (int i = 0; i < daughter_counter[iDecay]; i++) {
        int j                         = i + 1;
        id_KFDecayVertex[iDecay][j]   = daughters_KF[iDecay][i].Id();
        idMC_KFDecayVertex[iDecay][j] = daughters_MC[iDecay][i].trackID;
        pdg_KFDecayVertex[iDecay][j]  = daughters_KF[iDecay][i].GetPDG();
        for (int iPar = 0; iPar < 6; iPar++) {
          params_KFDecayVertex[iDecay][j][iPar] = daughters_KF[iDecay][i].GetParameter(iPar);
        }
        for (int iCov = 0; iCov < 36; iCov++) {
          covMat_KFDecayVertex[iDecay][j][iCov] = daughters_KF[iDecay][i].GetCovariance(iCov);
        }
        isCovMatOK_KFDecayVertex[iDecay][j] = CheckCovMat(covMat_KFDecayVertex[iDecay][j], stats);
        if (!isCovMatOK_KFDecayVertex[iDecay][j]) { isAllCovMatOK_KFDecayVertex[iDecay] = false; }
        x_KFDecayVertex[iDecay][j]        = daughters_KF[iDecay][i].GetX();
        y_KFDecayVertex[iDecay][j]        = daughters_KF[iDecay][i].GetY();
        z_KFDecayVertex[iDecay][j]        = daughters_KF[iDecay][i].GetZ();
        pX_KFDecayVertex[iDecay][j]       = daughters_KF[iDecay][i].GetPx();
        pY_KFDecayVertex[iDecay][j]       = daughters_KF[iDecay][i].GetPy();
        pZ_KFDecayVertex[iDecay][j]       = daughters_KF[iDecay][i].GetPz();
        e_KFDecayVertex[iDecay][j]        = daughters_KF[iDecay][i].GetE();
        s_KFDecayVertex[iDecay][j]        = daughters_KF[iDecay][i].GetS();
        charge_KFDecayVertex[iDecay][j]   = (int) daughters_KF[iDecay][i].GetQ();
        chi2_KFDecayVertex[iDecay][j]     = daughters_KF[iDecay][i].GetChi2();
        ndf_KFDecayVertex[iDecay][j]      = daughters_KF[iDecay][i].GetNDF();
        massHypo_KFDecayVertex[iDecay][j] = daughters_KF[iDecay][i].GetMassHypo();

        isMother_KFDecayVertex[iDecay][j]   = false;
        isDaughter_KFDecayVertex[iDecay][j] = true;

        daughters_KF[iDecay][i].GetP(p_KFDecayVertex[iDecay][j], pErr_KFDecayVertex[iDecay][j]);
        daughters_KF[iDecay][i].GetPt(pT_KFDecayVertex[iDecay][j], pTErr_KFDecayVertex[iDecay][j]);
        daughters_KF[iDecay][i].GetEta(eta_KFDecayVertex[iDecay][j], etaErr_KFDecayVertex[iDecay][j]);
        daughters_KF[iDecay][i].GetPhi(phi_KFDecayVertex[iDecay][j], phiErr_KFDecayVertex[iDecay][j]);
        daughters_KF[iDecay][i].GetMass(mass_KFDecayVertex[iDecay][j], massErr_KFDecayVertex[iDecay][j]);
        daughters_KF[iDecay][i].GetDecayLength(decayLength_KFDecayVertex[iDecay][j],
                                               decayLengthErr_KFDecayVertex[iDecay][j]);
        daughters_KF[iDecay][i].GetDecayLengthXY(decayLengthXY_KFDecayVertex[iDecay][j],
                                                 decayLengthXYErr_KFDecayVertex[iDecay][j]);
        daughters_KF[iDecay][i].GetLifeTime(lifeTime_KFDecayVertex[iDecay][j], lifeTimeErr_KFDecayVertex[iDecay][j]);
        daughters_KF[iDecay][i].GetR(r_KFDecayVertex[iDecay][j], rErr_KFDecayVertex[iDecay][j]);

        rapidity_KFDecayVertex[iDecay][j] = daughters_KF[iDecay][i].GetRapidity();
        theta_KFDecayVertex[iDecay][j]    = daughters_KF[iDecay][i].GetTheta();

        xErr_KFDecayVertex[iDecay][j]  = daughters_KF[iDecay][i].GetErrX();
        yErr_KFDecayVertex[iDecay][j]  = daughters_KF[iDecay][i].GetErrY();
        zErr_KFDecayVertex[iDecay][j]  = daughters_KF[iDecay][i].GetErrZ();
        pXErr_KFDecayVertex[iDecay][j] = daughters_KF[iDecay][i].GetErrPx();
        pYErr_KFDecayVertex[iDecay][j] = daughters_KF[iDecay][i].GetErrPy();
        pZErr_KFDecayVertex[iDecay][j] = daughters_KF[iDecay][i].GetErrPz();
        eErr_KFDecayVertex[iDecay][j]  = daughters_KF[iDecay][i].GetErrE();
        sErr_KFDecayVertex[iDecay][j]  = daughters_KF[iDecay][i].GetErrS();
      }
      treeKFDecayVertex[iDecay].Fill();
      treeKFProductionVertex[iDecay].Fill();
    }
  }  // end of event loop

  stats.PrintSummary(true);


  for (int iDecay = 0; iDecay < NUM_OF_DECAYS; iDecay++) {
    treeMCInitial[iDecay].Write();
    treeMCFinal[iDecay].Write();
    treeKFProductionVertex[iDecay].Write();
    treeKFDecayVertex[iDecay].Write();
  }

  outputFile.Close();


  cout << "The macro is done. Results are saved in " << KFTreeOutputFile << endl;
}
