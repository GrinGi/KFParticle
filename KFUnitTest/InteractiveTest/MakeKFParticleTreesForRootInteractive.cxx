#include "../UnitTests/ConfigConstants.h"
#include "../UnitTests/HeadersList.h"
#include "../UnitTests/ParticleStructures/MCParticleStructure.h"
#include "TFile.h"
#include "TString.h"
#include "TTree.h"


void MakeKFParticleTreesForRootInteractive()
{
#ifndef ALIPHYSICS
  gSystem->AddIncludePath(" -I$KFPARTICLE_DIR/install/usr/local/include");
  gSystem->Load("$KFPARTICLE_DIR/install/usr/local/lib/libKFParticle.dylib");
  gROOT->ProcessLine("KFParticle start_script_part3324151532");
#endif

  cout << "\n Starting MakeKFParticleTreesForRootInteractive macro \n";

  srand(time(NULL));

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
  KFParticle::SetField(magField[2] * 10);  // Bz, *10 to convert tesla to kgauss
  {
    float xyz[3] = {0, 0, 0};
    float B[3];
    KFParticle p;
//    p.GetFieldValue(xyz, B);
    std::cout << "Field is set to " << B[0] << " " << B[1] << " " << B[2] << " kilogaus" << std::endl;
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
  TString outputFileName = "RootInteractive_" + KFTreeOutputFile;
  TFile outputFile(outputFileName, "RECREATE");


  // ##### prepare KFARtree
  TTree treeKFAR;  // = new TTree("KFAfterReco", "KFAfterReco");
  treeKFAR.SetObject(nameOfKFARTree, nameOfKFARTree);

  int nTracks_KFAR;
  int numOfMCEvent_KFAR;
  bool isAllCovMatOK_KFAR;
  int constructMethod;

  int id_KFAR[NUM_OF_KFPARTICLES];
  int idMC_KFAR[NUM_OF_KFPARTICLES];
  int pdg_KFAR[NUM_OF_KFPARTICLES];
  float params_KFAR[NUM_OF_KFPARTICLES][6];
  float covMat_KFAR[NUM_OF_KFPARTICLES][36];
  bool isCovMatOK_KFAR[NUM_OF_KFPARTICLES];

  int massConstraint[NUM_OF_KFPARTICLES];
  int topoConstraint[NUM_OF_KFPARTICLES];

  float x_KFAR[NUM_OF_KFPARTICLES];
  float y_KFAR[NUM_OF_KFPARTICLES];
  float z_KFAR[NUM_OF_KFPARTICLES];
  float pX_KFAR[NUM_OF_KFPARTICLES];
  float pY_KFAR[NUM_OF_KFPARTICLES];
  float pZ_KFAR[NUM_OF_KFPARTICLES];
  float e_KFAR[NUM_OF_KFPARTICLES];
  float s_KFAR[NUM_OF_KFPARTICLES];
  int charge_KFAR[NUM_OF_KFPARTICLES];
  float chi2_KFAR[NUM_OF_KFPARTICLES];
  float ndf_KFAR[NUM_OF_KFPARTICLES];
  float massHypo_KFAR[NUM_OF_KFPARTICLES];

  bool isMother_KFAR[NUM_OF_KFPARTICLES];
  bool isDaughter_KFAR[NUM_OF_KFPARTICLES];

  float xErr_KFAR[NUM_OF_KFPARTICLES];
  float yErr_KFAR[NUM_OF_KFPARTICLES];
  float zErr_KFAR[NUM_OF_KFPARTICLES];
  float pXErr_KFAR[NUM_OF_KFPARTICLES];
  float pYErr_KFAR[NUM_OF_KFPARTICLES];
  float pZErr_KFAR[NUM_OF_KFPARTICLES];
  float p_KFAR[NUM_OF_KFPARTICLES];
  float pT_KFAR[NUM_OF_KFPARTICLES];
  float eta_KFAR[NUM_OF_KFPARTICLES];
  float phi_KFAR[NUM_OF_KFPARTICLES];
  float mass_KFAR[NUM_OF_KFPARTICLES];
  float decayLength_KFAR[NUM_OF_KFPARTICLES];
  float decayLengthXY_KFAR[NUM_OF_KFPARTICLES];
  float lifeTime_KFAR[NUM_OF_KFPARTICLES];
  float r_KFAR[NUM_OF_KFPARTICLES];
  float rapidity_KFAR[NUM_OF_KFPARTICLES];
  float theta_KFAR[NUM_OF_KFPARTICLES];

  float eErr_KFAR[NUM_OF_KFPARTICLES];
  float sErr_KFAR[NUM_OF_KFPARTICLES];
  float pErr_KFAR[NUM_OF_KFPARTICLES];
  float pTErr_KFAR[NUM_OF_KFPARTICLES];
  float etaErr_KFAR[NUM_OF_KFPARTICLES];
  float phiErr_KFAR[NUM_OF_KFPARTICLES];
  float massErr_KFAR[NUM_OF_KFPARTICLES];
  float decayLengthErr_KFAR[NUM_OF_KFPARTICLES];
  float decayLengthXYErr_KFAR[NUM_OF_KFPARTICLES];
  float lifeTimeErr_KFAR[NUM_OF_KFPARTICLES];
  float rErr_KFAR[NUM_OF_KFPARTICLES];

  treeKFAR.Branch("nTracks", &nTracks_KFAR, "nTracks/I");
  treeKFAR.Branch("numOfMCEvent", &numOfMCEvent_KFAR, "numOfMCEvent/I");
  treeKFAR.Branch("constructMethod", &constructMethod, "constructMethod/I");
  treeKFAR.Branch("isAllCovMatOK", &isAllCovMatOK_KFAR, "isAllCovMatOK/O");

  treeKFAR.Branch("ID", id_KFAR, "ID[nTracks]/I");
  treeKFAR.Branch("IDMC", idMC_KFAR, "IDMC[nTracks]/I");
  treeKFAR.Branch("pdgID", pdg_KFAR, "pdgID[nTracks]/I");
  treeKFAR.Branch("params", &params_KFAR, "params[nTracks][6]/F");
  treeKFAR.Branch("covMat", &covMat_KFAR, "covMat[nTracks][36]/F");
  treeKFAR.Branch("isCovMatOK", &isCovMatOK_KFAR, "isCovMatOK[nTracks]/O");

  treeKFAR.Branch("massConstraint", massConstraint, "massConstraint[nTracks]/I");
  treeKFAR.Branch("topoConstraint", topoConstraint, "topoConstraint[nTracks]/I");

  treeKFAR.Branch("x", &x_KFAR, "x[nTracks]/F");
  treeKFAR.Branch("y", &y_KFAR, "y[nTracks]/F");
  treeKFAR.Branch("z", &z_KFAR, "z[nTracks]/F");
  treeKFAR.Branch("pX", &pX_KFAR, "pX[nTracks]/F");
  treeKFAR.Branch("pY", &pY_KFAR, "pY[nTracks]/F");
  treeKFAR.Branch("pZ", &pZ_KFAR, "pZ[nTracks]/F");
  treeKFAR.Branch("e", &e_KFAR, "e[nTracks]/F");
  treeKFAR.Branch("s", &s_KFAR, "s[nTracks]/F");
  treeKFAR.Branch("charge", &charge_KFAR, "charge[nTracks]/I");
  treeKFAR.Branch("chi2", &chi2_KFAR, "chi2[nTracks]/F");
  treeKFAR.Branch("ndf", &ndf_KFAR, "ndf[nTracks]/F");
  treeKFAR.Branch("massHypo", &massHypo_KFAR, "massHypo[nTracks]/F");

  treeKFAR.Branch("isMother", &isMother_KFAR, "isMother[nTracks]/O");
  treeKFAR.Branch("isDaughter", &isDaughter_KFAR, "isDaughter[nTracks]/O");

  treeKFAR.Branch("p", &p_KFAR, "p[nTracks]/F");
  treeKFAR.Branch("pT", &pT_KFAR, "pT[nTracks]/F");
  treeKFAR.Branch("eta", &eta_KFAR, "eta[nTracks]/F");
  treeKFAR.Branch("phi", &phi_KFAR, "phi[nTracks]/F");
  treeKFAR.Branch("mass", &mass_KFAR, "mass[nTracks]/F");
  treeKFAR.Branch("decayLength", &decayLength_KFAR, "decayLength[nTracks]/F");
  treeKFAR.Branch("decayLengthXY", &decayLengthXY_KFAR, "decayLengthXY[nTracks]/F");
  treeKFAR.Branch("lifeTime", &lifeTime_KFAR, "lifeTime[nTracks]/F");
  treeKFAR.Branch("r", &r_KFAR, "r[nTracks]/F");
  treeKFAR.Branch("rapidity", &rapidity_KFAR, "rapidity[nTracks]/F");
  treeKFAR.Branch("theta", &theta_KFAR, "theta[nTracks]/F");

  treeKFAR.Branch("xErr", &xErr_KFAR, "xErr[nTracks]/F");
  treeKFAR.Branch("yErr", &yErr_KFAR, "yErr[nTracks]/F");
  treeKFAR.Branch("zErr", &zErr_KFAR, "zErr[nTracks]/F");
  treeKFAR.Branch("pXErr", &pXErr_KFAR, "pXErr[nTracks]/F");
  treeKFAR.Branch("pYErr", &pYErr_KFAR, "pYErr[nTracks]/F");
  treeKFAR.Branch("pZErr", &pZErr_KFAR, "pZErr[nTracks]/F");
  treeKFAR.Branch("eErr", &eErr_KFAR, "eErr[nTracks]/F");
  treeKFAR.Branch("sErr", &sErr_KFAR, "sErr[nTracks]/F");
  treeKFAR.Branch("pErr", &pErr_KFAR, "pErr[nTracks]/F");
  treeKFAR.Branch("pTErr", &pTErr_KFAR, "pTErr[nTracks]/F");
  treeKFAR.Branch("etaErr", &etaErr_KFAR, "etaErr[nTracks]/F");
  treeKFAR.Branch("phiErr", &phiErr_KFAR, "phiErr[nTracks]/F");
  treeKFAR.Branch("massErr", &massErr_KFAR, "massErr[nTracks]/F");
  treeKFAR.Branch("decayLengthErr", &decayLengthErr_KFAR, "decayLengthErr[nTracks]/F");
  treeKFAR.Branch("decayLengthXYErr", &decayLengthXYErr_KFAR, "decayLengthXYErr[nTracks]/F");
  treeKFAR.Branch("lifeTimeErr", &lifeTimeErr_KFAR, "lifeTimeErr[nTracks]/F");
  treeKFAR.Branch("rErr", &rErr_KFAR, "rErr[nTracks]/F");


  int nEvents = chain.GetEntries();

  cout << endl;
  cout << "Starting the event loop\n";
  ////////////////////////////////////////////////////////
  //          START OF THE EVENT LOOP
  ////////////////////////////////////////////////////////
  for (int iEvent = 0; iEvent < nEvents; iEvent++) {
    chain.GetEntry(iEvent);

    // print a processing progress
    if (iEvent % 1000 == 0) {
      std::cout << "processing " << iEvent << " event\r";
      std::cout << std::flush;
    }

    // if number of tracks in the event less than
    // number of daughters + 1 (mother particle) - do not process the event
    if (nTracks_MC < NUM_OF_MOTHERS + NUM_OF_DAUGHTERS) {
      std::cout << "Event # " << iEvent << " has " << nTracks_MC
                << ", which is less than (NUM_OF_MOTHERS + NUM_OF_DAUGHTERS). Skip this event." << std::endl;
      continue;
    }

    // create MC particles structure
    MCParticleStruct mother_MC;
    std::array<MCParticleStruct, NUM_OF_DAUGHTERS> daughters_MC;

    // create KFParticle structure for particles
    KFParticle mother_KF;
    std::array<KFParticle, NUM_OF_DAUGHTERS> daughters_KF;

    size_t motherCounter   = 0;
    size_t daughterCounter = 0;

    // loop over the tracks
    for (int iTrack = 0; iTrack < nTracks_MC; iTrack++) {
      // if we found all daughters and mothers - do not process other tracks
      if (daughterCounter >= NUM_OF_DAUGHTERS && motherCounter >= NUM_OF_MOTHERS) { break; }
      //setup mother particle from MC data
      //mother_MC.ndf = ?????
      if (parentID_MC[iTrack] == 0) {
        mother_MC.trackID  = iTrack;  //trackID_MC[iTrack];
        mother_MC.parentID = parentID_MC[iTrack];
        mother_MC.pdg      = pdg_MC[iTrack];
        //mother_MC.pdgLifeTime = pdgLifeTime_MC[iTrack];
        mother_MC.charge      = charge_MC[iTrack];
        mother_MC.isMother    = true;
        mother_MC.initialX    = initialX_MC[iTrack];
        mother_MC.initialY    = initialY_MC[iTrack];
        mother_MC.initialZ    = initialZ_MC[iTrack];
        mother_MC.finalX      = finalX_MC[iTrack];
        mother_MC.finalY      = finalY_MC[iTrack];
        mother_MC.finalZ      = finalZ_MC[iTrack];
        mother_MC.trackLength = trackLength_MC[iTrack];
        mother_MC.initialPX   = initialPX_MC[iTrack];
        mother_MC.initialPY   = initialPY_MC[iTrack];
        mother_MC.initialPZ   = initialPZ_MC[iTrack];
        mother_MC.finalPX     = finalPX_MC[iTrack];
        mother_MC.finalPY     = finalPY_MC[iTrack];
        mother_MC.finalPZ     = finalPZ_MC[iTrack];
        mother_MC.mass        = mass_MC[iTrack];
        motherCounter++;
      }
      if (parentID_MC[iTrack] == 1) {
        daughters_MC[daughterCounter].trackID  = iTrack;  //trackID_MC[iTrack];
        daughters_MC[daughterCounter].parentID = parentID_MC[iTrack];
        daughters_MC[daughterCounter].pdg      = pdg_MC[iTrack];
        //daughters_MC[daughterCounter].pdgLifeTime = pdgLifeTime_MC[iTrack];
        daughters_MC[daughterCounter].charge      = charge_MC[iTrack];
        daughters_MC[daughterCounter].isMother    = false;
        daughters_MC[daughterCounter].initialX    = initialX_MC[iTrack];
        daughters_MC[daughterCounter].initialY    = initialY_MC[iTrack];
        daughters_MC[daughterCounter].initialZ    = initialZ_MC[iTrack];
        daughters_MC[daughterCounter].finalX      = finalX_MC[iTrack];
        daughters_MC[daughterCounter].finalY      = finalY_MC[iTrack];
        daughters_MC[daughterCounter].finalZ      = finalZ_MC[iTrack];
        daughters_MC[daughterCounter].trackLength = trackLength_MC[iTrack];
        daughters_MC[daughterCounter].initialPX   = initialPX_MC[iTrack];
        daughters_MC[daughterCounter].initialPY   = initialPY_MC[iTrack];
        daughters_MC[daughterCounter].initialPZ   = initialPZ_MC[iTrack];
        daughters_MC[daughterCounter].finalPX     = finalPX_MC[iTrack];
        daughters_MC[daughterCounter].finalPY     = finalPY_MC[iTrack];
        daughters_MC[daughterCounter].finalPZ     = finalPZ_MC[iTrack];
        daughters_MC[daughterCounter].mass        = mass_MC[iTrack];

        daughterCounter++;
      }
    }
    // end of track loop

    //processing...
    if (!(motherCounter == NUM_OF_MOTHERS && daughterCounter == NUM_OF_DAUGHTERS)) {
      std::cout << "Warning! In the event # " << iEvent
                << " motherCounter < NUM_OF_MOTHERS or daughterCounter < NUM_OF_DAUGHTERS. Do not taking the event "
                   "into account.\n";
      continue;
    }

    /// Create KFParticles from MCParticleStructs
    float params_M[6]              = {mother_MC.finalX,  mother_MC.finalY,  mother_MC.finalZ,
                                      mother_MC.finalPX, mother_MC.finalPY, mother_MC.finalPZ};
    std::vector<float> covmat_vec_ = MakeCovMatrix(mother_MC);
    float covmat_M[21];
    std::copy(covmat_vec_.begin(), covmat_vec_.end(), covmat_M);
    int charge_ = mother_MC.charge;
    float chi2_ = mother_MC.chi2;
    int ndf_    = mother_MC.ndf;
    float mass_ = mother_MC.mass;
    mother_KF.SetId(mother_MC.trackID);
    mother_KF.SetPDG(mother_MC.pdg);

    for (int i = 0; i < NUM_OF_DAUGHTERS; i++) {
      float params_D[6] = {daughters_MC[i].initialX,  daughters_MC[i].initialY,  daughters_MC[i].initialZ,
                           daughters_MC[i].initialPX, daughters_MC[i].initialPY, daughters_MC[i].initialPZ};
      covmat_vec_       = MakeCovMatrix(daughters_MC[i]);
      float covmat_D[21];
      std::copy(covmat_vec_.begin(), covmat_vec_.end(), covmat_D);
      charge_ = daughters_MC[i].charge;
      chi2_   = 0;  //daughters_MC[i].chi2;
      ndf_    = 0;  //daughters_MC[i].ndf;
      mass_   = daughters_MC[i].mass;
#ifdef TESTSUITE
      daughters_KF[i].Create(params_D, covmat_D, charge_, chi2_, ndf_, mass_);
#else
      daughters_KF[i].Create(params_D, covmat_D, charge_, mass_);
#endif
      daughters_KF[i].SetId(daughters_MC[i].trackID);
      daughters_KF[i].SetPDG(daughters_MC[i].pdg);
    }

    // do particle reconstruction here
    constructMethod = rand() % 2;  // 0 or 2 ;
    if (constructMethod == 1) { constructMethod = 2; }
    mother_KF.SetConstructMethod(constructMethod);

    bool isDecomposedOK = true;
    for (auto& part : daughters_KF) {
      isDecomposedOK = SmearParticle(part);
      //cout << iEvent << "\n";
      mother_KF.AddDaughter(part);
    }

    int topoConstrMother = rand() % 2;  //0 for no, 1 for yes
    topoConstraint[0]    = topoConstrMother;
    if (topoConstrMother == 1) {
      KFPVertex vert;
      vert.SetXYZ(mother_MC.initialX, mother_MC.initialY, mother_MC.initialZ);
      //vert.SetCovarianceMatrix(1.,0.,1.,0.,0.,1.); // ????
      vert.SetCovarianceMatrix(mother_KF.GetCovariance(0), mother_KF.GetCovariance(1), mother_KF.GetCovariance(2),
                               mother_KF.GetCovariance(3), mother_KF.GetCovariance(4),
                               mother_KF.GetCovariance(5));  // ????
#ifdef MAKE_VERTEX_COV_MAT
      std::vector<float> covmat_vec_vertex_ = MakeVertexCovMatrix(mother_MC);
      float covmat_vec_vertex_array_[6];
      std::copy(covmat_vec_vertex_.begin(), covmat_vec_vertex_.end(), covmat_vec_vertex_array_);
      vert.SetCovarianceMatrix(covmat_vec_vertex_array_);
      SmearVertex(vert);
#endif
      float decay_point[] = {mother_KF.GetX(), mother_KF.GetY(), mother_KF.GetZ()};
      mother_KF.SetProductionVertex(KFVertex(vert));
      mother_KF.TransportToPoint(decay_point);
    }
    else {
      ///
    }

    int massConstrMother = rand() % 3;  // 0 for no, 1 for linear, 2 for nonlinear
    massConstraint[0]    = massConstrMother;
    if (massConstrMother == 1) { mother_KF.SetMassConstraint(mother_MC.mass); }
    else if (massConstrMother == 2) {
      mother_KF.SetNonlinearMassConstraint(mother_MC.mass);
    }
    else {
      ///
    }

    int topoConstrDaughters = rand() % 2;  //0 for no, 1 for yes
    for (int i = 0; i < NUM_OF_DAUGHTERS; i++) {
      int j             = i + NUM_OF_MOTHERS;
      topoConstraint[j] = topoConstrDaughters;
    }
    if (topoConstrDaughters == 1) {
      for (auto& part : daughters_KF) { part.SetProductionVertex(mother_KF); }
    }
    else {
      ///
    }

    int massConstrDaughters = rand() % 3;  // 0 for no, 1 for linear, 2 for nonlinear
    if (massConstrDaughters != 0 && massConstrMother != 0) { massConstrDaughters = massConstrMother; }
    for (int i = 0; i < NUM_OF_DAUGHTERS; i++) {
      int j             = i + NUM_OF_MOTHERS;
      massConstraint[j] = massConstrDaughters;
    }
    if (massConstrDaughters == 1) {
      for (auto& part : daughters_KF) { part.SetMassConstraint(part.GetMassHypo()); }
    }
    else if (massConstrDaughters == 2) {
      for (auto& part : daughters_KF) { part.SetNonlinearMassConstraint(part.GetMassHypo()); }
    }
    else {
      ///
    }

    // Fill KFAR tree
    nTracks_KFAR       = NUM_OF_MOTHERS + NUM_OF_DAUGHTERS;
    numOfMCEvent_KFAR  = iEvent;
    isAllCovMatOK_KFAR = isDecomposedOK;

    id_KFAR[0]   = mother_KF.Id();
    idMC_KFAR[0] = mother_MC.trackID;
    pdg_KFAR[0]  = mother_KF.GetPDG();
    for (int iPar = 0; iPar < 6; iPar++) { params_KFAR[0][iPar] = mother_KF.GetParameter(iPar); }
    for (int iCov = 0; iCov < 36; iCov++) { covMat_KFAR[0][iCov] = mother_KF.GetCovariance(iCov); }
    isCovMatOK_KFAR[0] = CheckCovMat(covMat_KFAR[0]);
    if (!isCovMatOK_KFAR[0]) { isAllCovMatOK_KFAR = false; }
    x_KFAR[0]        = mother_KF.GetX();
    y_KFAR[0]        = mother_KF.GetY();
    z_KFAR[0]        = mother_KF.GetZ();
    pX_KFAR[0]       = mother_KF.GetPx();
    pY_KFAR[0]       = mother_KF.GetPy();
    pZ_KFAR[0]       = mother_KF.GetPz();
    e_KFAR[0]        = mother_KF.GetE();
    s_KFAR[0]        = mother_KF.GetS();
    charge_KFAR[0]   = (int) mother_KF.GetQ();
    chi2_KFAR[0]     = mother_KF.GetChi2();
    ndf_KFAR[0]      = mother_KF.GetNDF();
    massHypo_KFAR[0] = mother_KF.GetMassHypo();

    isMother_KFAR[0]   = true;
    isDaughter_KFAR[0] = false;

    mother_KF.GetP(p_KFAR[0], pErr_KFAR[0]);
    mother_KF.GetPt(pT_KFAR[0], pTErr_KFAR[0]);
    mother_KF.GetEta(eta_KFAR[0], etaErr_KFAR[0]);
    mother_KF.GetPhi(phi_KFAR[0], phiErr_KFAR[0]);
    mother_KF.GetMass(mass_KFAR[0], massErr_KFAR[0]);
    mother_KF.GetDecayLength(decayLength_KFAR[0], decayLengthErr_KFAR[0]);
    mother_KF.GetDecayLengthXY(decayLengthXY_KFAR[0], decayLengthXYErr_KFAR[0]);
    mother_KF.GetLifeTime(lifeTime_KFAR[0], lifeTimeErr_KFAR[0]);
    mother_KF.GetR(r_KFAR[0], rErr_KFAR[0]);

    rapidity_KFAR[0] = mother_KF.GetRapidity();
    theta_KFAR[0]    = mother_KF.GetTheta();

    xErr_KFAR[0]  = mother_KF.GetErrX();
    yErr_KFAR[0]  = mother_KF.GetErrY();
    zErr_KFAR[0]  = mother_KF.GetErrZ();
    pXErr_KFAR[0] = mother_KF.GetErrPx();
    pYErr_KFAR[0] = mother_KF.GetErrPy();
    pZErr_KFAR[0] = mother_KF.GetErrPz();
    eErr_KFAR[0]  = mother_KF.GetErrE();
    sErr_KFAR[0]  = mother_KF.GetErrS();

    for (int i = 0; i < NUM_OF_DAUGHTERS; i++) {
      int j        = i + NUM_OF_MOTHERS;
      id_KFAR[j]   = daughters_KF[i].Id();
      idMC_KFAR[j] = daughters_MC[i].trackID;
      pdg_KFAR[j]  = daughters_KF[i].GetPDG();
      for (int iPar = 0; iPar < 6; iPar++) { params_KFAR[j][iPar] = daughters_KF[i].GetParameter(iPar); }
      for (int iCov = 0; iCov < 36; iCov++) { covMat_KFAR[j][iCov] = daughters_KF[i].GetCovariance(iCov); }
      isCovMatOK_KFAR[j] = CheckCovMat(covMat_KFAR[j]);
      if (!isCovMatOK_KFAR[j]) { isAllCovMatOK_KFAR = false; }
      x_KFAR[j]        = daughters_KF[i].GetX();
      y_KFAR[j]        = daughters_KF[i].GetY();
      z_KFAR[j]        = daughters_KF[i].GetZ();
      pX_KFAR[j]       = daughters_KF[i].GetPx();
      pY_KFAR[j]       = daughters_KF[i].GetPy();
      pZ_KFAR[j]       = daughters_KF[i].GetPz();
      e_KFAR[j]        = daughters_KF[i].GetE();
      s_KFAR[j]        = daughters_KF[i].GetS();
      charge_KFAR[j]   = (int) daughters_KF[i].GetQ();
      chi2_KFAR[j]     = daughters_KF[i].GetChi2();
      ndf_KFAR[j]      = daughters_KF[i].GetNDF();
      massHypo_KFAR[j] = daughters_KF[i].GetMassHypo();

      isMother_KFAR[j]   = false;
      isDaughter_KFAR[j] = true;

      daughters_KF[i].GetP(p_KFAR[j], pErr_KFAR[j]);
      daughters_KF[i].GetPt(pT_KFAR[j], pTErr_KFAR[j]);
      daughters_KF[i].GetEta(eta_KFAR[j], etaErr_KFAR[j]);
      daughters_KF[i].GetPhi(phi_KFAR[j], phiErr_KFAR[j]);
      daughters_KF[i].GetMass(mass_KFAR[j], massErr_KFAR[j]);
      daughters_KF[i].GetDecayLength(decayLength_KFAR[j], decayLengthErr_KFAR[j]);
      daughters_KF[i].GetDecayLengthXY(decayLengthXY_KFAR[j], decayLengthXYErr_KFAR[j]);
      daughters_KF[i].GetLifeTime(lifeTime_KFAR[j], lifeTimeErr_KFAR[j]);
      daughters_KF[i].GetR(r_KFAR[j], rErr_KFAR[j]);

      rapidity_KFAR[j] = daughters_KF[i].GetRapidity();
      theta_KFAR[j]    = daughters_KF[i].GetTheta();

      xErr_KFAR[j]  = daughters_KF[i].GetErrX();
      yErr_KFAR[j]  = daughters_KF[i].GetErrY();
      zErr_KFAR[j]  = daughters_KF[i].GetErrZ();
      pXErr_KFAR[j] = daughters_KF[i].GetErrPx();
      pYErr_KFAR[j] = daughters_KF[i].GetErrPy();
      pZErr_KFAR[j] = daughters_KF[i].GetErrPz();
      eErr_KFAR[j]  = daughters_KF[i].GetErrE();
      sErr_KFAR[j]  = daughters_KF[i].GetErrS();
    }

    treeKFAR.Fill();
  }
  // end of event loop


  treeKFAR.Write();

  outputFile.Close();

  cout << "The macro is done. Results are saved in " << KFTreeOutputFile << endl;
}

///////
/////////
///////////////
