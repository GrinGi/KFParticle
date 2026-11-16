#include "ConfigConstants.h"
#include "HeadersList.h"
#include "OutputObjInterface/MakeGraphsFromTH2.h"
#include "OutputObjInterface/OutputMap.h"
#include "TDirectory.h"
#include "TH1.h"

#include <iostream>
#include <algorithm>

/////
float pT(float px, float py) { return sqrt(px * px + py * py); }
float p(float px, float py, float pz) { return sqrt(px * px + py * py + pz * pz); }
float e(float px, float py, float pz, float mass) { return sqrt(px * px + py * py + pz * pz + mass * mass); }
float eta(float px, float py, float pz) { return 0.5 * log((p(px, py, pz) + pz) / (p(px, py, pz) - pz)); }
float phi(float px, float py) { return atan2(py, px); }
float theta(float px, float py, float pz) { return atan2(pT(px, py), pz); }

// ... ADD DESCRIPTION
//...

void MakeKFUnitTestHistos()
{

  cout << "\n Staring MakeKFUnitTestHistos macro \n";

  TFile outputFile(KFHistoOutputFile, "RECREATE");

  OutputMap output_map(outputFile);
  output_map.CreateAll();

  ///////////////////////////////////////////////////////////
  // Next step is to create a directories structure
  //////////////////////////////////////////////////////////

  TDirectory* QAHistos_Dir = outputFile.mkdir("QAHistos");

  TDirectory* GeantMC_Dir = QAHistos_Dir->mkdir("GeantMC");
  std::array<TDirectory*, NUM_OF_DECAYS> DecayDir;

  std::array<TDirectory*, NUM_OF_DECAYS> MCInitialDir;
  std::array<TDirectory*, NUM_OF_DECAYS> MotherMCInitial_Dir;
  std::array<std::array<TDirectory*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> DaughtersMCInitial_Dir;

  std::array<TDirectory*, NUM_OF_DECAYS> MCFinalDir;
  std::array<TDirectory*, NUM_OF_DECAYS> MotherMCFinal_Dir;
  std::array<std::array<TDirectory*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> DaughtersMCFinal_Dir;

  std::array<TDirectory*, NUM_OF_DECAYS> KFProductionVertexDir;
  std::array<TDirectory*, NUM_OF_DECAYS> MotherKFProductionVertex_Dir;
  std::array<std::array<TDirectory*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> DaughtersKFProductionVertex_Dir;

  std::array<TDirectory*, NUM_OF_DECAYS> KFDecayVertexDir;
  std::array<TDirectory*, NUM_OF_DECAYS> MotherKFDecayVertex_Dir;
  std::array<std::array<TDirectory*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> DaughtersKFDecayVertex_Dir;

  for (int iDecay = 0; iDecay < NUM_OF_DECAYS; iDecay++) {
    DecayDir[iDecay] = QAHistos_Dir->mkdir(Form("%i", MOTHER_PDG_IDS[iDecay]));

    MCInitialDir[iDecay]        = DecayDir[iDecay]->mkdir("MCInitial");
    MotherMCInitial_Dir[iDecay] = MCInitialDir[iDecay]->mkdir(Form("Mother_%i", MOTHER_PDG_IDS[iDecay]));
    for (int i = 0; i < NUM_OF_DAUGHTERS[iDecay]; i++) {
      DaughtersMCInitial_Dir[iDecay][i] =
        MCInitialDir[iDecay]->mkdir(Form("Daughter%i_%i", i + 1, DAUGHTER_PDG_IDS[iDecay][i]));
    }

    MCFinalDir[iDecay]        = DecayDir[iDecay]->mkdir("MCFinal");
    MotherMCFinal_Dir[iDecay] = MCFinalDir[iDecay]->mkdir(Form("Mother_%i", MOTHER_PDG_IDS[iDecay]));
    for (int i = 0; i < NUM_OF_DAUGHTERS[iDecay]; i++) {
      DaughtersMCFinal_Dir[iDecay][i] =
        MCFinalDir[iDecay]->mkdir(Form("Daughter%i_%i", i + 1, DAUGHTER_PDG_IDS[iDecay][i]));
    }

    KFProductionVertexDir[iDecay] = DecayDir[iDecay]->mkdir("KFPartBeforeReco");
    MotherKFProductionVertex_Dir[iDecay] =
      KFProductionVertexDir[iDecay]->mkdir(Form("Mother_%i", MOTHER_PDG_IDS[iDecay]));
    for (int i = 0; i < NUM_OF_DAUGHTERS[iDecay]; i++) {
      DaughtersKFProductionVertex_Dir[iDecay][i] =
        KFProductionVertexDir[iDecay]->mkdir(Form("Daughter%i_%i", i + 1, DAUGHTER_PDG_IDS[iDecay][i]));
    }

    KFDecayVertexDir[iDecay]        = DecayDir[iDecay]->mkdir("KFPartAfterReco");
    MotherKFDecayVertex_Dir[iDecay] = KFDecayVertexDir[iDecay]->mkdir(Form("Mother_%i", MOTHER_PDG_IDS[iDecay]));
    for (int i = 0; i < NUM_OF_DAUGHTERS[iDecay]; i++) {
      DaughtersKFDecayVertex_Dir[iDecay][i] =
        KFDecayVertexDir[iDecay]->mkdir(Form("Daughter%i_%i", i + 1, DAUGHTER_PDG_IDS[iDecay][i]));
    }
  }

  //TDirectory*  = outputFile.mkdir("OtherHistosAndPlots");
  TDirectory* OtherHistos = outputFile.mkdir("OtherHistosAndPlots");


  ////////////////////////////////////////////////////////
  ////////////// end of structure block
  ////////////////////////////////////////////////////////

  ////////////////////////////////////////////////////////
  ////////////// Beginning of histograms block
  ////////////////////////////////////////////////////////

  std::array<TH1F*, NUM_OF_DECAYS> fHistXMotherMCInitial;
  std::array<TH1F*, NUM_OF_DECAYS> fHistYMotherMCInitial;
  std::array<TH1F*, NUM_OF_DECAYS> fHistZMotherMCInitial;
  std::array<TH1F*, NUM_OF_DECAYS> fHistPxMotherMCInitial;
  std::array<TH1F*, NUM_OF_DECAYS> fHistPyMotherMCInitial;
  std::array<TH1F*, NUM_OF_DECAYS> fHistPzMotherMCInitial;
  std::array<TH1I*, NUM_OF_DECAYS> fHistChargeMotherMCInitial;
  std::array<TH1I*, NUM_OF_DECAYS> fHistPDGMotherMCInitial;
  std::array<TH1F*, NUM_OF_DECAYS> fHistPtMotherMCInitial;
  std::array<TH1F*, NUM_OF_DECAYS> fHistOneOverPtMotherMCInitial;
  std::array<TH1F*, NUM_OF_DECAYS> fHistPMotherMCInitial;
  std::array<TH1F*, NUM_OF_DECAYS> fHistEMotherMCInitial;
  std::array<TH1F*, NUM_OF_DECAYS> fHistMassMotherMCInitial;
  std::array<TH1F*, NUM_OF_DECAYS> fHistEtaMotherMCInitial;
  std::array<TH1F*, NUM_OF_DECAYS> fHistPhiMotherMCInitial;

  std::array<std::array<TH1F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistXDaughterMCInitial;
  std::array<std::array<TH1F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistYDaughterMCInitial;
  std::array<std::array<TH1F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistZDaughterMCInitial;
  std::array<std::array<TH1F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistPxDaughterMCInitial;
  std::array<std::array<TH1F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistPyDaughterMCInitial;
  std::array<std::array<TH1F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistPzDaughterMCInitial;
  std::array<std::array<TH1I*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistChargeDaughterMCInitial;
  std::array<std::array<TH1I*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistPDGDaughterMCInitial;
  std::array<std::array<TH1F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistPtDaughterMCInitial;
  std::array<std::array<TH1F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistOneOverPtDaughterMCInitial;
  std::array<std::array<TH1F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistPDaughterMCInitial;
  std::array<std::array<TH1F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistEDaughterMCInitial;
  std::array<std::array<TH1F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistMassDaughterMCInitial;
  std::array<std::array<TH1F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistEtaDaughterMCInitial;
  std::array<std::array<TH1F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistPhiDaughterMCInitial;

  for (int iDecay = 0; iDecay < NUM_OF_DECAYS; iDecay++) {
    fHistXMotherMCInitial[iDecay] =
      new TH1F(Form("fHistXMotherMCInitial_%i", MOTHER_PDG_IDS[iDecay]), "X MC; x, cm; counts", 1000, -1e1, 1e1);
    fHistYMotherMCInitial[iDecay] =
      new TH1F(Form("fHistYMotherMCInitial_%i", MOTHER_PDG_IDS[iDecay]), "Y MC; y, cm; counts", 1000, -1e1, 1e1);
    fHistZMotherMCInitial[iDecay] =
      new TH1F(Form("fHistZMotherMCInitial_%i", MOTHER_PDG_IDS[iDecay]), "Z MC; z, cm; counts", 1000, -3e1, 3e1);
    fHistPxMotherMCInitial[iDecay] =
      new TH1F(Form("fHistPxMotherMCInitial_%i", MOTHER_PDG_IDS[iDecay]), "Px MC; Px, GeV; counts", 1000, -10, 10);
    fHistPyMotherMCInitial[iDecay] =
      new TH1F(Form("fHistPyMotherMCInitial_%i", MOTHER_PDG_IDS[iDecay]), "Py MC; Py, GeV; counts", 1000, -10, 10);
    fHistPzMotherMCInitial[iDecay] =
      new TH1F(Form("fHistPzMotherMCInitial_%i", MOTHER_PDG_IDS[iDecay]), "Pz MC; Pz, GeV; counts", 1000, -10, 10);
    fHistChargeMotherMCInitial[iDecay] =
      new TH1I(Form("fHistChargeMotherMCInitial_%i", MOTHER_PDG_IDS[iDecay]), "Charge MC; Q; counts", 50, -25, 25);
    fHistPDGMotherMCInitial[iDecay] =
      new TH1I(Form("fHistPDGMotherMCInitial_%i", MOTHER_PDG_IDS[iDecay]), "PDG MC; PDG; counts", 2000, -1000, 1000);
    fHistPtMotherMCInitial[iDecay] =
      new TH1F(Form("fHistPtMotherMCInitial_%i", MOTHER_PDG_IDS[iDecay]), "Pt MC; p_{T}, GeV; counts", 1000, 0., 10.);
    fHistOneOverPtMotherMCInitial[iDecay] = new TH1F(Form("fHistOveOverPtMotherMCInitial_%i", MOTHER_PDG_IDS[iDecay]),
                                                     "1/Pt MC; 1/Pt GeV^{-1}", 1000, 0., 20.);
    fHistPMotherMCInitial[iDecay] =
      new TH1F(Form("fHistPMotherMCInitial_%i", MOTHER_PDG_IDS[iDecay]), "P MC; P, GeV; counts", 1000, 0., 30.);
    fHistEMotherMCInitial[iDecay] =
      new TH1F(Form("fHistEMotherMCInitial_%i", MOTHER_PDG_IDS[iDecay]), "E MC; E, GeV; counts", 1000, 0., 30.);
    fHistMassMotherMCInitial[iDecay] =
      new TH1F(Form("fHistMassMotherMCInitial_%i", MOTHER_PDG_IDS[iDecay]), "Mass MC; M, GeV; counts", 1000, 0., 10.);
    fHistEtaMotherMCInitial[iDecay] =
      new TH1F(Form("fHistEtaMotherMCInitial_%i", MOTHER_PDG_IDS[iDecay]), "Eta MC; #eta; counts", 1000, -10, 10);
    fHistPhiMotherMCInitial[iDecay] =
      new TH1F(Form("fHistPhiMotherMCInitial_%i", MOTHER_PDG_IDS[iDecay]), "Phi MC; #varphi; counts", 1000, -6.5, 6.5);
    for (int iDaughter = 0; iDaughter < NUM_OF_DAUGHTERS[iDecay]; iDaughter++) {
      fHistXDaughterMCInitial[iDecay][iDaughter] =
        new TH1F(Form("fHistXDaughterMCInitial_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1), "X MC; x, cm; counts",
                 1000, -1e1, 1e1);
      fHistYDaughterMCInitial[iDecay][iDaughter] =
        new TH1F(Form("fHistYDaughterMCInitial_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1), "Y MC; y, cm; counts",
                 1000, -1e1, 1e1);
      fHistZDaughterMCInitial[iDecay][iDaughter] =
        new TH1F(Form("fHistZDaughterMCInitial_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1), "Z MC; z, cm; counts",
                 1000, -3e1, 3e1);
      fHistPxDaughterMCInitial[iDecay][iDaughter] =
        new TH1F(Form("fHistPxDaughterMCInitial_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1),
                 "Px MC; Px, GeV; counts", 1000, -10, 10);
      fHistPyDaughterMCInitial[iDecay][iDaughter] =
        new TH1F(Form("fHistPyDaughterMCInitial_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1),
                 "Py MC; Py, GeV; counts", 1000, -10, 10);
      fHistPzDaughterMCInitial[iDecay][iDaughter] =
        new TH1F(Form("fHistPzDaughterMCInitial_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1),
                 "Pz MC; Pz, GeV; counts", 1000, -10, 10);
      fHistChargeDaughterMCInitial[iDecay][iDaughter] =
        new TH1I(Form("fHistChargeDaughterMCInitial_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1),
                 "Charge MC; Q; counts", 50, -25, 25);
      fHistPDGDaughterMCInitial[iDecay][iDaughter] =
        new TH1I(Form("fHistPDGDaughterMCInitial_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1), "PDG MC; PDG; counts",
                 2000, -1000, 1000);
      fHistPtDaughterMCInitial[iDecay][iDaughter] =
        new TH1F(Form("fHistPtDaughterMCInitial_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1),
                 "Pt MC; p_{T}, GeV; counts", 1000, 0., 10.);
      fHistOneOverPtDaughterMCInitial[iDecay][iDaughter] =
        new TH1F(Form("fHistOveOverPtDaughterMCInitial_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1),
                 "1/Pt MC; 1/Pt GeV^{-1}", 1000, 0., 20.);
      fHistPDaughterMCInitial[iDecay][iDaughter] =
        new TH1F(Form("fHistPDaughterMCInitial_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1), "P MC; P, GeV; counts",
                 1000, 0., 30.);
      fHistEDaughterMCInitial[iDecay][iDaughter] =
        new TH1F(Form("fHistEDaughterMCInitial_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1), "E MC; E, GeV; counts",
                 1000, 0., 30.);
      fHistMassDaughterMCInitial[iDecay][iDaughter] =
        new TH1F(Form("fHistMassDaughterMCInitial_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1),
                 "Mass MC; M, GeV; counts", 1000, 0., 10.);
      fHistEtaDaughterMCInitial[iDecay][iDaughter] =
        new TH1F(Form("fHistEtaDaughterMCInitial_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1), "Eta MC; #eta; counts",
                 1000, -10, 10.);
      fHistPhiDaughterMCInitial[iDecay][iDaughter] =
        new TH1F(Form("fHistPhiDaughterMCInitial_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1),
                 "Phi MC; #varphi; counts", 1000, -6.5, 6.5);
    }
  }

  std::array<TH1F*, NUM_OF_DECAYS> fHistXMotherMCFinal;
  std::array<TH1F*, NUM_OF_DECAYS> fHistYMotherMCFinal;
  std::array<TH1F*, NUM_OF_DECAYS> fHistZMotherMCFinal;
  std::array<TH1F*, NUM_OF_DECAYS> fHistPxMotherMCFinal;
  std::array<TH1F*, NUM_OF_DECAYS> fHistPyMotherMCFinal;
  std::array<TH1F*, NUM_OF_DECAYS> fHistPzMotherMCFinal;
  std::array<TH1I*, NUM_OF_DECAYS> fHistChargeMotherMCFinal;
  std::array<TH1I*, NUM_OF_DECAYS> fHistPDGMotherMCFinal;
  std::array<TH1F*, NUM_OF_DECAYS> fHistPtMotherMCFinal;
  std::array<TH1F*, NUM_OF_DECAYS> fHistOneOverPtMotherMCFinal;
  std::array<TH1F*, NUM_OF_DECAYS> fHistPMotherMCFinal;
  std::array<TH1F*, NUM_OF_DECAYS> fHistEMotherMCFinal;
  std::array<TH1F*, NUM_OF_DECAYS> fHistMassMotherMCFinal;
  std::array<TH1F*, NUM_OF_DECAYS> fHistEtaMotherMCFinal;
  std::array<TH1F*, NUM_OF_DECAYS> fHistPhiMotherMCFinal;

  std::array<std::array<TH1F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistXDaughterMCFinal;
  std::array<std::array<TH1F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistYDaughterMCFinal;
  std::array<std::array<TH1F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistZDaughterMCFinal;
  std::array<std::array<TH1F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistPxDaughterMCFinal;
  std::array<std::array<TH1F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistPyDaughterMCFinal;
  std::array<std::array<TH1F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistPzDaughterMCFinal;
  std::array<std::array<TH1I*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistChargeDaughterMCFinal;
  std::array<std::array<TH1I*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistPDGDaughterMCFinal;
  std::array<std::array<TH1F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistPtDaughterMCFinal;
  std::array<std::array<TH1F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistOneOverPtDaughterMCFinal;
  std::array<std::array<TH1F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistPDaughterMCFinal;
  std::array<std::array<TH1F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistEDaughterMCFinal;
  std::array<std::array<TH1F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistMassDaughterMCFinal;
  std::array<std::array<TH1F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistEtaDaughterMCFinal;
  std::array<std::array<TH1F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistPhiDaughterMCFinal;

  for (int iDecay = 0; iDecay < NUM_OF_DECAYS; iDecay++) {
    fHistXMotherMCFinal[iDecay] =
      new TH1F(Form("fHistXMotherMCFinal_%i", MOTHER_PDG_IDS[iDecay]), "X MC; x, cm; counts", 1000, -1e1, 1e1);
    fHistYMotherMCFinal[iDecay] =
      new TH1F(Form("fHistYMotherMCFinal_%i", MOTHER_PDG_IDS[iDecay]), "Y MC; y, cm; counts", 1000, -1e1, 1e1);
    fHistZMotherMCFinal[iDecay] =
      new TH1F(Form("fHistZMotherMCFinal_%i", MOTHER_PDG_IDS[iDecay]), "Z MC; z, cm; counts", 1000, -3e1, 3e1);
    fHistPxMotherMCFinal[iDecay] =
      new TH1F(Form("fHistPxMotherMCFinal_%i", MOTHER_PDG_IDS[iDecay]), "Px MC; Px, GeV; counts", 1000, -10, 10);
    fHistPyMotherMCFinal[iDecay] =
      new TH1F(Form("fHistPyMotherMCFinal_%i", MOTHER_PDG_IDS[iDecay]), "Py MC; Py, GeV; counts", 1000, -10, 10);
    fHistPzMotherMCFinal[iDecay] =
      new TH1F(Form("fHistPzMotherMCFinal_%i", MOTHER_PDG_IDS[iDecay]), "Pz MC; Pz, GeV; counts", 1000, -10, 10);
    fHistChargeMotherMCFinal[iDecay] =
      new TH1I(Form("fHistChargeMotherMCFinal_%i", MOTHER_PDG_IDS[iDecay]), "Charge MC; Q; counts", 50, -25, 25);
    fHistPDGMotherMCFinal[iDecay] =
      new TH1I(Form("fHistPDGMotherMCFinal_%i", MOTHER_PDG_IDS[iDecay]), "PDG MC; PDG; counts", 2000, -1000, 1000);
    fHistPtMotherMCFinal[iDecay] =
      new TH1F(Form("fHistPtMotherMCFinal_%i", MOTHER_PDG_IDS[iDecay]), "Pt MC; p_{T}, GeV; counts", 1000, 0., 10.);
    fHistOneOverPtMotherMCFinal[iDecay] =
      new TH1F(Form("fHistOveOverPtMotherMCFinal_%i", MOTHER_PDG_IDS[iDecay]), "1/Pt MC; 1/Pt GeV^{-1}", 1000, 0., 20.);
    fHistPMotherMCFinal[iDecay] =
      new TH1F(Form("fHistPMotherMCFinal_%i", MOTHER_PDG_IDS[iDecay]), "P MC; P, GeV; counts", 1000, 0., 30.);
    fHistEMotherMCFinal[iDecay] =
      new TH1F(Form("fHistEMotherMCFinal_%i", MOTHER_PDG_IDS[iDecay]), "E MC; E, GeV; counts", 1000, 0., 30.);
    fHistMassMotherMCFinal[iDecay] =
      new TH1F(Form("fHistMassMotherMCFinal_%i", MOTHER_PDG_IDS[iDecay]), "Mass MC; M, GeV; counts", 1000, 0., 10.);
    fHistEtaMotherMCFinal[iDecay] =
      new TH1F(Form("fHistEtaMotherMCFinal_%i", MOTHER_PDG_IDS[iDecay]), "Eta MC; #eta; counts", 1000, -10, 10);
    fHistPhiMotherMCFinal[iDecay] =
      new TH1F(Form("fHistPhiMotherMCFinal_%i", MOTHER_PDG_IDS[iDecay]), "Phi MC; #varphi; counts", 1000, -6.5, 6.5);
    for (int iDaughter = 0; iDaughter < NUM_OF_DAUGHTERS[iDecay]; iDaughter++) {
      fHistXDaughterMCFinal[iDecay][iDaughter] =
        new TH1F(Form("fHistXDaughterMCFinal_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1), "X MC; x, cm; counts",
                 1000, -1e1, 1e1);
      fHistYDaughterMCFinal[iDecay][iDaughter] =
        new TH1F(Form("fHistYDaughterMCFinal_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1), "Y MC; y, cm; counts",
                 1000, -1e1, 1e1);
      fHistZDaughterMCFinal[iDecay][iDaughter] =
        new TH1F(Form("fHistZDaughterMCFinal_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1), "Z MC; z, cm; counts",
                 1000, -3e1, 3e1);
      fHistPxDaughterMCFinal[iDecay][iDaughter] =
        new TH1F(Form("fHistPxDaughterMCFinal_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1), "Px MC; Px, GeV; counts",
                 1000, -10, 10);
      fHistPyDaughterMCFinal[iDecay][iDaughter] =
        new TH1F(Form("fHistPyDaughterMCFinal_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1), "Py MC; Py, GeV; counts",
                 1000, -10, 10);
      fHistPzDaughterMCFinal[iDecay][iDaughter] =
        new TH1F(Form("fHistPzDaughterMCFinal_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1), "Pz MC; Pz, GeV; counts",
                 1000, -10, 10);
      fHistChargeDaughterMCFinal[iDecay][iDaughter] =
        new TH1I(Form("fHistChargeDaughterMCFinal_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1),
                 "Charge MC; Q; counts", 50, -25, 25);
      fHistPDGDaughterMCFinal[iDecay][iDaughter] =
        new TH1I(Form("fHistPDGDaughterMCFinal_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1), "PDG MC; PDG; counts",
                 2000, -1000, 1000);
      fHistPtDaughterMCFinal[iDecay][iDaughter] =
        new TH1F(Form("fHistPtDaughterMCFinal_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1),
                 "Pt MC; p_{T}, GeV; counts", 1000, 0., 10.);
      fHistOneOverPtDaughterMCFinal[iDecay][iDaughter] =
        new TH1F(Form("fHistOveOverPtDaughterMCFinal_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1),
                 "1/Pt MC; 1/Pt GeV^{-1}", 1000, 0., 20.);
      fHistPDaughterMCFinal[iDecay][iDaughter] =
        new TH1F(Form("fHistPDaughterMCFinal_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1), "P MC; P, GeV; counts",
                 1000, 0., 30.);
      fHistEDaughterMCFinal[iDecay][iDaughter] =
        new TH1F(Form("fHistEDaughterMCFinal_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1), "E MC; E, GeV; counts",
                 1000, 0., 30.);
      fHistMassDaughterMCFinal[iDecay][iDaughter] =
        new TH1F(Form("fHistMassDaughterMCFinal_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1),
                 "Mass MC; M, GeV; counts", 1000, 0., 10.);
      fHistEtaDaughterMCFinal[iDecay][iDaughter] =
        new TH1F(Form("fHistEtaDaughterMCFinal_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1), "Eta MC; #eta; counts",
                 1000, -10, 10.);
      fHistPhiDaughterMCFinal[iDecay][iDaughter] =
        new TH1F(Form("fHistPhiDaughterMCFinal_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1),
                 "Phi MC; #varphi; counts", 1000, -6.5, 6.5);
    }
  }

  std::array<TH1F*, NUM_OF_DECAYS> fHistXMotherKFProductionVertex;
  std::array<TH1F*, NUM_OF_DECAYS> fHistYMotherKFProductionVertex;
  std::array<TH1F*, NUM_OF_DECAYS> fHistZMotherKFProductionVertex;
  std::array<TH1F*, NUM_OF_DECAYS> fHistPxMotherKFProductionVertex;
  std::array<TH1F*, NUM_OF_DECAYS> fHistPyMotherKFProductionVertex;
  std::array<TH1F*, NUM_OF_DECAYS> fHistPzMotherKFProductionVertex;
  std::array<TH1I*, NUM_OF_DECAYS> fHistChargeMotherKFProductionVertex;
  std::array<TH1I*, NUM_OF_DECAYS> fHistPDGMotherKFProductionVertex;
  std::array<TH1F*, NUM_OF_DECAYS> fHistPtMotherKFProductionVertex;
  std::array<TH1F*, NUM_OF_DECAYS> fHistOneOverPtMotherKFProductionVertex;
  std::array<TH1F*, NUM_OF_DECAYS> fHistPMotherKFProductionVertex;
  std::array<TH1F*, NUM_OF_DECAYS> fHistEMotherKFProductionVertex;
  std::array<TH1F*, NUM_OF_DECAYS> fHistMassMotherKFProductionVertex;
  std::array<TH1F*, NUM_OF_DECAYS> fHistEtaMotherKFProductionVertex;
  std::array<TH1F*, NUM_OF_DECAYS> fHistPhiMotherKFProductionVertex;
  std::array<TH1F*, NUM_OF_DECAYS> fHistChi2MotherKFProductionVertex;
  std::array<TH1F*, NUM_OF_DECAYS> fHistNDFMotherKFProductionVertex;

  std::array<std::array<TH1F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistXDaughterKFProductionVertex;
  std::array<std::array<TH1F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistYDaughterKFProductionVertex;
  std::array<std::array<TH1F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistZDaughterKFProductionVertex;
  std::array<std::array<TH1F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistPxDaughterKFProductionVertex;
  std::array<std::array<TH1F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistPyDaughterKFProductionVertex;
  std::array<std::array<TH1F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistPzDaughterKFProductionVertex;
  std::array<std::array<TH1I*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistChargeDaughterKFProductionVertex;
  std::array<std::array<TH1I*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistPDGDaughterKFProductionVertex;
  std::array<std::array<TH1F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistPtDaughterKFProductionVertex;
  std::array<std::array<TH1F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistOneOverPtDaughterKFProductionVertex;
  std::array<std::array<TH1F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistPDaughterKFProductionVertex;
  std::array<std::array<TH1F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistEDaughterKFProductionVertex;
  std::array<std::array<TH1F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistMassDaughterKFProductionVertex;
  std::array<std::array<TH1F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistEtaDaughterKFProductionVertex;
  std::array<std::array<TH1F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistPhiDaughterKFProductionVertex;
  std::array<std::array<TH1F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistChi2DaughterKFProductionVertex;
  std::array<std::array<TH1F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistNDFDaughterKFProductionVertex;

  for (int iDecay = 0; iDecay < NUM_OF_DECAYS; iDecay++) {
    fHistXMotherKFProductionVertex[iDecay] = new TH1F(Form("fHistXMotherKFProductionVertex_%i", MOTHER_PDG_IDS[iDecay]),
                                                      "X MC; x, cm; counts", 1000, -1e1, 1e1);
    fHistYMotherKFProductionVertex[iDecay] = new TH1F(Form("fHistYMotherKFProductionVertex_%i", MOTHER_PDG_IDS[iDecay]),
                                                      "Y MC; y, cm; counts", 1000, -1e1, 1e1);
    fHistZMotherKFProductionVertex[iDecay] = new TH1F(Form("fHistZMotherKFProductionVertex_%i", MOTHER_PDG_IDS[iDecay]),
                                                      "Z MC; z, cm; counts", 1000, -3e1, 3e1);
    fHistPxMotherKFProductionVertex[iDecay] = new TH1F(
      Form("fHistPxMotherKFProductionVertex_%i", MOTHER_PDG_IDS[iDecay]), "Px MC; Px, GeV; counts", 1000, -10, 10);
    fHistPyMotherKFProductionVertex[iDecay] = new TH1F(
      Form("fHistPyMotherKFProductionVertex_%i", MOTHER_PDG_IDS[iDecay]), "Py MC; Py, GeV; counts", 1000, -10, 10);
    fHistPzMotherKFProductionVertex[iDecay] = new TH1F(
      Form("fHistPzMotherKFProductionVertex_%i", MOTHER_PDG_IDS[iDecay]), "Pz MC; Pz, GeV; counts", 1000, -10, 10);
    fHistChargeMotherKFProductionVertex[iDecay] = new TH1I(
      Form("fHistChargeMotherKFProductionVertex_%i", MOTHER_PDG_IDS[iDecay]), "Charge MC; Q; counts", 50, -25, 25);
    fHistPDGMotherKFProductionVertex[iDecay] = new TH1I(
      Form("fHistPDGMotherKFProductionVertex_%i", MOTHER_PDG_IDS[iDecay]), "PDG MC; PDG; counts", 2000, -1000, 1000);
    fHistPtMotherKFProductionVertex[iDecay] = new TH1F(
      Form("fHistPtMotherKFProductionVertex_%i", MOTHER_PDG_IDS[iDecay]), "Pt MC; p_{T}, GeV; counts", 1000, 0., 10.);
    fHistOneOverPtMotherKFProductionVertex[iDecay] =
      new TH1F(Form("fHistOveOverPtMotherKFProductionVertex_%i", MOTHER_PDG_IDS[iDecay]), "1/Pt MC; 1/Pt GeV^{-1}",
               1000, 0., 20.);
    fHistPMotherKFProductionVertex[iDecay] = new TH1F(Form("fHistPMotherKFProductionVertex_%i", MOTHER_PDG_IDS[iDecay]),
                                                      "P MC; P, GeV; counts", 1000, 0., 30.);
    fHistEMotherKFProductionVertex[iDecay] = new TH1F(Form("fHistEMotherKFProductionVertex_%i", MOTHER_PDG_IDS[iDecay]),
                                                      "E MC; E, GeV; counts", 1000, 0., 30.);
    fHistMassMotherKFProductionVertex[iDecay] = new TH1F(
      Form("fHistMassMotherKFProductionVertex_%i", MOTHER_PDG_IDS[iDecay]), "Mass MC; M, GeV; counts", 1000, 0., 10.);
    fHistEtaMotherKFProductionVertex[iDecay] = new TH1F(
      Form("fHistEtaMotherKFProductionVertex_%i", MOTHER_PDG_IDS[iDecay]), "Eta MC; #eta; counts", 1000, -10, 10);
    fHistPhiMotherKFProductionVertex[iDecay] = new TH1F(
      Form("fHistPhiMotherKFProductionVertex_%i", MOTHER_PDG_IDS[iDecay]), "Phi MC; #varphi; counts", 1000, -6.5, 6.5);
    fHistChi2MotherKFProductionVertex[iDecay] = new TH1F(
      Form("fHistChi2MotherKFProductionVertex_%i", MOTHER_PDG_IDS[iDecay]), "Chi2 MC; #chi^{2}; counts", 1000, -10, 10);
    fHistNDFMotherKFProductionVertex[iDecay] =
      new TH1F(Form("fHistNDFMotherKFProductionVertex_%i", MOTHER_PDG_IDS[iDecay]), "NDF MC; NDF; counts", 14, -2, 12);

    for (int iDaughter = 0; iDaughter < NUM_OF_DAUGHTERS[iDecay]; iDaughter++) {
      fHistXDaughterKFProductionVertex[iDecay][iDaughter] =
        new TH1F(Form("fHistXDaughterKFProductionVertex_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1),
                 "X MC; x, cm; counts", 1000, -1e1, 1e1);
      fHistYDaughterKFProductionVertex[iDecay][iDaughter] =
        new TH1F(Form("fHistYDaughterKFProductionVertex_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1),
                 "Y MC; y, cm; counts", 1000, -1e1, 1e1);
      fHistZDaughterKFProductionVertex[iDecay][iDaughter] =
        new TH1F(Form("fHistZDaughterKFProductionVertex_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1),
                 "Z MC; z, cm; counts", 1000, -3e1, 3e1);
      fHistPxDaughterKFProductionVertex[iDecay][iDaughter] =
        new TH1F(Form("fHistPxDaughterKFProductionVertex_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1),
                 "Px MC; Px, GeV; counts", 1000, -10, 10);
      fHistPyDaughterKFProductionVertex[iDecay][iDaughter] =
        new TH1F(Form("fHistPyDaughterKFProductionVertex_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1),
                 "Py MC; Py, GeV; counts", 1000, -10, 10);
      fHistPzDaughterKFProductionVertex[iDecay][iDaughter] =
        new TH1F(Form("fHistPzDaughterKFProductionVertex_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1),
                 "Pz MC; Pz, GeV; counts", 1000, -10, 10);
      fHistChargeDaughterKFProductionVertex[iDecay][iDaughter] =
        new TH1I(Form("fHistChargeDaughterKFProductionVertex_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1),
                 "Charge MC; Q; counts", 50, -25, 25);
      fHistPDGDaughterKFProductionVertex[iDecay][iDaughter] =
        new TH1I(Form("fHistPDGDaughterKFProductionVertex_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1),
                 "PDG MC; PDG; counts", 2000, -1000, 1000);
      fHistPtDaughterKFProductionVertex[iDecay][iDaughter] =
        new TH1F(Form("fHistPtDaughterKFProductionVertex_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1),
                 "Pt MC; p_{T}, GeV; counts", 1000, 0., 10.);
      fHistOneOverPtDaughterKFProductionVertex[iDecay][iDaughter] =
        new TH1F(Form("fHistOveOverPtDaughterKFProductionVertex_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1),
                 "1/Pt MC; 1/Pt GeV^{-1}", 1000, 0., 20.);
      fHistPDaughterKFProductionVertex[iDecay][iDaughter] =
        new TH1F(Form("fHistPDaughterKFProductionVertex_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1),
                 "P MC; P, GeV; counts", 1000, 0., 30.);
      fHistEDaughterKFProductionVertex[iDecay][iDaughter] =
        new TH1F(Form("fHistEDaughterKFProductionVertex_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1),
                 "E MC; E, GeV; counts", 1000, 0., 30.);
      fHistMassDaughterKFProductionVertex[iDecay][iDaughter] =
        new TH1F(Form("fHistMassDaughterKFProductionVertex_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1),
                 "Mass MC; M, GeV; counts", 1000, 0., 10.);
      fHistEtaDaughterKFProductionVertex[iDecay][iDaughter] =
        new TH1F(Form("fHistEtaDaughterKFProductionVertex_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1),
                 "Eta MC; #eta; counts", 1000, -10, 10.);
      fHistPhiDaughterKFProductionVertex[iDecay][iDaughter] =
        new TH1F(Form("fHistPhiDaughterKFProductionVertex_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1),
                 "Phi MC; #varphi; counts", 1000, -6.5, 6.5);
      fHistChi2DaughterKFProductionVertex[iDecay][iDaughter] =
        new TH1F(Form("fHistChi2DaughterKFProductionVertex_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1),
                 "Chi2 MC; #chi^{2}; counts", 1000, -10, 10);
      fHistNDFDaughterKFProductionVertex[iDecay][iDaughter] =
        new TH1F(Form("fHistNDFDaughterKFProductionVertex_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1),
                 "NDF MC; NDF; counts", 14, -2, 12);
    }
  }

  std::array<TH1F*, NUM_OF_DECAYS> fHistXMotherKFDecayVertex;
  std::array<TH1F*, NUM_OF_DECAYS> fHistYMotherKFDecayVertex;
  std::array<TH1F*, NUM_OF_DECAYS> fHistZMotherKFDecayVertex;
  std::array<TH1F*, NUM_OF_DECAYS> fHistPxMotherKFDecayVertex;
  std::array<TH1F*, NUM_OF_DECAYS> fHistPyMotherKFDecayVertex;
  std::array<TH1F*, NUM_OF_DECAYS> fHistPzMotherKFDecayVertex;
  std::array<TH1I*, NUM_OF_DECAYS> fHistChargeMotherKFDecayVertex;
  std::array<TH1I*, NUM_OF_DECAYS> fHistPDGMotherKFDecayVertex;
  std::array<TH1F*, NUM_OF_DECAYS> fHistPtMotherKFDecayVertex;
  std::array<TH1F*, NUM_OF_DECAYS> fHistOneOverPtMotherKFDecayVertex;
  std::array<TH1F*, NUM_OF_DECAYS> fHistPMotherKFDecayVertex;
  std::array<TH1F*, NUM_OF_DECAYS> fHistEMotherKFDecayVertex;
  std::array<TH1F*, NUM_OF_DECAYS> fHistMassMotherKFDecayVertex;
  std::array<TH1F*, NUM_OF_DECAYS> fHistEtaMotherKFDecayVertex;
  std::array<TH1F*, NUM_OF_DECAYS> fHistPhiMotherKFDecayVertex;
  std::array<TH1F*, NUM_OF_DECAYS> fHistChi2MotherKFDecayVertex;
  std::array<TH1F*, NUM_OF_DECAYS> fHistNDFMotherKFDecayVertex;

  std::array<std::array<TH1F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistXDaughterKFDecayVertex;
  std::array<std::array<TH1F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistYDaughterKFDecayVertex;
  std::array<std::array<TH1F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistZDaughterKFDecayVertex;
  std::array<std::array<TH1F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistPxDaughterKFDecayVertex;
  std::array<std::array<TH1F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistPyDaughterKFDecayVertex;
  std::array<std::array<TH1F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistPzDaughterKFDecayVertex;
  std::array<std::array<TH1I*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistChargeDaughterKFDecayVertex;
  std::array<std::array<TH1I*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistPDGDaughterKFDecayVertex;
  std::array<std::array<TH1F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistPtDaughterKFDecayVertex;
  std::array<std::array<TH1F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistOneOverPtDaughterKFDecayVertex;
  std::array<std::array<TH1F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistPDaughterKFDecayVertex;
  std::array<std::array<TH1F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistEDaughterKFDecayVertex;
  std::array<std::array<TH1F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistMassDaughterKFDecayVertex;
  std::array<std::array<TH1F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistEtaDaughterKFDecayVertex;
  std::array<std::array<TH1F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistPhiDaughterKFDecayVertex;
  std::array<std::array<TH1F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistChi2DaughterKFDecayVertex;
  std::array<std::array<TH1F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistNDFDaughterKFDecayVertex;

  for (int iDecay = 0; iDecay < NUM_OF_DECAYS; iDecay++) {
    fHistXMotherKFDecayVertex[iDecay] =
      new TH1F(Form("fHistXMotherKFDecayVertex_%i", MOTHER_PDG_IDS[iDecay]), "X MC; x, cm; counts", 1000, -1e1, 1e1);
    fHistYMotherKFDecayVertex[iDecay] =
      new TH1F(Form("fHistYMotherKFDecayVertex_%i", MOTHER_PDG_IDS[iDecay]), "Y MC; y, cm; counts", 1000, -1e1, 1e1);
    fHistZMotherKFDecayVertex[iDecay] =
      new TH1F(Form("fHistZMotherKFDecayVertex_%i", MOTHER_PDG_IDS[iDecay]), "Z MC; z, cm; counts", 1000, -3e1, 3e1);
    fHistPxMotherKFDecayVertex[iDecay] =
      new TH1F(Form("fHistPxMotherKFDecayVertex_%i", MOTHER_PDG_IDS[iDecay]), "Px MC; Px, GeV; counts", 1000, -10, 10);
    fHistPyMotherKFDecayVertex[iDecay] =
      new TH1F(Form("fHistPyMotherKFDecayVertex_%i", MOTHER_PDG_IDS[iDecay]), "Py MC; Py, GeV; counts", 1000, -10, 10);
    fHistPzMotherKFDecayVertex[iDecay] =
      new TH1F(Form("fHistPzMotherKFDecayVertex_%i", MOTHER_PDG_IDS[iDecay]), "Pz MC; Pz, GeV; counts", 1000, -10, 10);
    fHistChargeMotherKFDecayVertex[iDecay] =
      new TH1I(Form("fHistChargeMotherKFDecayVertex_%i", MOTHER_PDG_IDS[iDecay]), "Charge MC; Q; counts", 50, -25, 25);
    fHistPDGMotherKFDecayVertex[iDecay]       = new TH1I(Form("fHistPDGMotherKFDecayVertex_%i", MOTHER_PDG_IDS[iDecay]),
                                                         "PDG MC; PDG; counts", 2000, -1000, 1000);
    fHistPtMotherKFDecayVertex[iDecay]        = new TH1F(Form("fHistPtMotherKFDecayVertex_%i", MOTHER_PDG_IDS[iDecay]),
                                                         "Pt MC; p_{T}, GeV; counts", 1000, 0., 10.);
    fHistOneOverPtMotherKFDecayVertex[iDecay] = new TH1F(
      Form("fHistOveOverPtMotherKFDecayVertex_%i", MOTHER_PDG_IDS[iDecay]), "1/Pt MC; 1/Pt GeV^{-1}", 1000, 0., 20.);
    fHistPMotherKFDecayVertex[iDecay] =
      new TH1F(Form("fHistPMotherKFDecayVertex_%i", MOTHER_PDG_IDS[iDecay]), "P MC; P, GeV; counts", 1000, 0., 30.);
    fHistEMotherKFDecayVertex[iDecay] =
      new TH1F(Form("fHistEMotherKFDecayVertex_%i", MOTHER_PDG_IDS[iDecay]), "E MC; E, GeV; counts", 1000, 0., 30.);
    fHistMassMotherKFDecayVertex[iDecay] = new TH1F(Form("fHistMassMotherKFDecayVertex_%i", MOTHER_PDG_IDS[iDecay]),
                                                    "Mass MC; M, GeV; counts", 1000, 0., 10.);
    fHistEtaMotherKFDecayVertex[iDecay] =
      new TH1F(Form("fHistEtaMotherKFDecayVertex_%i", MOTHER_PDG_IDS[iDecay]), "Eta MC; #eta; counts", 1000, -10, 10);
    fHistPhiMotherKFDecayVertex[iDecay]  = new TH1F(Form("fHistPhiMotherKFDecayVertex_%i", MOTHER_PDG_IDS[iDecay]),
                                                    "Phi MC; #varphi; counts", 1000, -6.5, 6.5);
    fHistChi2MotherKFDecayVertex[iDecay] = new TH1F(Form("fHistChi2MotherKFDecayVertex_%i", MOTHER_PDG_IDS[iDecay]),
                                                    "Chi2 MC; #chi^{2}; counts", 1000, -10, 10);
    fHistNDFMotherKFDecayVertex[iDecay] =
      new TH1F(Form("fHistNDFMotherKFDecayVertex_%i", MOTHER_PDG_IDS[iDecay]), "NDF MC; NDF; counts", 14, -2, 12);
    for (int iDaughter = 0; iDaughter < NUM_OF_DAUGHTERS[iDecay]; iDaughter++) {
      fHistXDaughterKFDecayVertex[iDecay][iDaughter] =
        new TH1F(Form("fHistXDaughterKFDecayVertex_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1),
                 "X MC; x, cm; counts", 1000, -1e1, 1e1);
      fHistYDaughterKFDecayVertex[iDecay][iDaughter] =
        new TH1F(Form("fHistYDaughterKFDecayVertex_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1),
                 "Y MC; y, cm; counts", 1000, -1e1, 1e1);
      fHistZDaughterKFDecayVertex[iDecay][iDaughter] =
        new TH1F(Form("fHistZDaughterKFDecayVertex_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1),
                 "Z MC; z, cm; counts", 1000, -3e1, 3e1);
      fHistPxDaughterKFDecayVertex[iDecay][iDaughter] =
        new TH1F(Form("fHistPxDaughterKFDecayVertex_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1),
                 "Px MC; Px, GeV; counts", 1000, -10, 10);
      fHistPyDaughterKFDecayVertex[iDecay][iDaughter] =
        new TH1F(Form("fHistPyDaughterKFDecayVertex_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1),
                 "Py MC; Py, GeV; counts", 1000, -10, 10);
      fHistPzDaughterKFDecayVertex[iDecay][iDaughter] =
        new TH1F(Form("fHistPzDaughterKFDecayVertex_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1),
                 "Pz MC; Pz, GeV; counts", 1000, -10, 10);
      fHistChargeDaughterKFDecayVertex[iDecay][iDaughter] =
        new TH1I(Form("fHistChargeDaughterKFDecayVertex_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1),
                 "Charge MC; Q; counts", 50, -25, 25);
      fHistPDGDaughterKFDecayVertex[iDecay][iDaughter] =
        new TH1I(Form("fHistPDGDaughterKFDecayVertex_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1),
                 "PDG MC; PDG; counts", 2000, -1000, 1000);
      fHistPtDaughterKFDecayVertex[iDecay][iDaughter] =
        new TH1F(Form("fHistPtDaughterKFDecayVertex_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1),
                 "Pt MC; p_{T}, GeV; counts", 1000, 0., 10.);
      fHistOneOverPtDaughterKFDecayVertex[iDecay][iDaughter] =
        new TH1F(Form("fHistOveOverPtDaughterKFDecayVertex_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1),
                 "1/Pt MC; 1/Pt GeV^{-1}", 1000, 0., 20.);
      fHistPDaughterKFDecayVertex[iDecay][iDaughter] =
        new TH1F(Form("fHistPDaughterKFDecayVertex_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1),
                 "P MC; P, GeV; counts", 1000, 0., 30.);
      fHistEDaughterKFDecayVertex[iDecay][iDaughter] =
        new TH1F(Form("fHistEDaughterKFDecayVertex_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1),
                 "E MC; E, GeV; counts", 1000, 0., 30.);
      fHistMassDaughterKFDecayVertex[iDecay][iDaughter] =
        new TH1F(Form("fHistMassDaughterKFDecayVertex_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1),
                 "Mass MC; M, GeV; counts", 1000, 0., 10.);
      fHistEtaDaughterKFDecayVertex[iDecay][iDaughter] =
        new TH1F(Form("fHistEtaDaughterKFDecayVertex_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1),
                 "Eta MC; #eta; counts", 1000, -10, 10.);
      fHistPhiDaughterKFDecayVertex[iDecay][iDaughter] =
        new TH1F(Form("fHistPhiDaughterKFDecayVertex_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1),
                 "Phi MC; #varphi; counts", 1000, -6.5, 6.5);
      fHistChi2DaughterKFDecayVertex[iDecay][iDaughter] =
        new TH1F(Form("fHistChi2DaughterKFDecayVertex_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1),
                 "Chi2 MC; #chi^{2}; counts", 1000, -10, 10);
      fHistNDFDaughterKFDecayVertex[iDecay][iDaughter] =
        new TH1F(Form("fHistNDFDaughterKFDecayVertex_%i_%i", MOTHER_PDG_IDS[iDecay], iDaughter + 1),
                 "NDF MC; NDF; counts", 14, -2, 12);
    }
  }

  // other histos and plots
  TH2F* fHistPtPtErrMother = new TH2F("fHistPtPtErrMother", "fHistPtPtErrMother", 50, 0, 10, 100, -0.04, 0.04);
  ////////////////////////////////////////////////////////////////////////
  ////////////////// end of histogram block
  ////////////////////////////////////////////////////////////////////////

  cout << " Creating input and ouput files \n";
  ////////////////////////////////////////////////////////////////////////
  ////////////////// read trees from file
  ////////////////////////////////////////////////////////////////////////
  TChain* chainMC = new TChain(nameOfMCTree);
  std::cout << "MCInputFile: " << MCInputFile << std::endl;
  chainMC->Add(MCInputFile);

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

  chainMC->SetBranchAddress("nTracks", &nTracks_MC);
  chainMC->SetBranchAddress("trackID", trackID_MC);
  chainMC->SetBranchAddress("parentID", parentID_MC);
  chainMC->SetBranchAddress("pdg", pdg_MC);
  //chainMC-> SetBranchAddress( "pdgLifeTime", pdgLifeTime_MC);
  chainMC->SetBranchAddress("charge", charge_MC);
  chainMC->SetBranchAddress("initialX", initialX_MC);
  chainMC->SetBranchAddress("initialY", initialY_MC);
  chainMC->SetBranchAddress("initialZ", initialZ_MC);
  chainMC->SetBranchAddress("finalX", finalX_MC);
  chainMC->SetBranchAddress("finalY", finalY_MC);
  chainMC->SetBranchAddress("finalZ", finalZ_MC);
  chainMC->SetBranchAddress("trackLength", trackLength_MC);
  chainMC->SetBranchAddress("initialPX", initialPX_MC);
  chainMC->SetBranchAddress("initialPY", initialPY_MC);
  chainMC->SetBranchAddress("initialPZ", initialPZ_MC);
  chainMC->SetBranchAddress("finalPX", finalPX_MC);
  chainMC->SetBranchAddress("finalPY", finalPY_MC);
  chainMC->SetBranchAddress("finalPZ", finalPZ_MC);
  chainMC->SetBranchAddress("mass", mass_MC);

  //--------------------------------------------------------------
  //Read MC Initial tree
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

  TChain* chainMCInitial[NUM_OF_DECAYS];

  for (int iDecay = 0; iDecay < NUM_OF_DECAYS; iDecay++) {
    chainMCInitial[iDecay] = new TChain(nameOfMCInitialOutputTree[iDecay]);


    chainMCInitial[iDecay]->Add(KFInputFile);

    chainMCInitial[iDecay]->SetBranchAddress("nTracks", &nTracks_MCInitial[iDecay]);
    chainMCInitial[iDecay]->SetBranchAddress("numOfMCEvent", &numOfMCEvent_MCInitial[iDecay]);

    chainMCInitial[iDecay]->SetBranchAddress("ID", id_MCInitial[iDecay]);
    chainMCInitial[iDecay]->SetBranchAddress("ParentID", parentid_MCInitial[iDecay]);
    chainMCInitial[iDecay]->SetBranchAddress("pdgID", pdg_MCInitial[iDecay]);
    chainMCInitial[iDecay]->SetBranchAddress("x", x_MCInitial[iDecay]);
    chainMCInitial[iDecay]->SetBranchAddress("y", y_MCInitial[iDecay]);
    chainMCInitial[iDecay]->SetBranchAddress("z", z_MCInitial[iDecay]);
    chainMCInitial[iDecay]->SetBranchAddress("pX", pX_MCInitial[iDecay]);
    chainMCInitial[iDecay]->SetBranchAddress("pY", pY_MCInitial[iDecay]);
    chainMCInitial[iDecay]->SetBranchAddress("pZ", pZ_MCInitial[iDecay]);
    chainMCInitial[iDecay]->SetBranchAddress("e", e_MCInitial[iDecay]);
    chainMCInitial[iDecay]->SetBranchAddress("s", s_MCInitial[iDecay]);
    chainMCInitial[iDecay]->SetBranchAddress("charge", charge_MCInitial[iDecay]);

    chainMCInitial[iDecay]->SetBranchAddress("isMother", isMother_MCInitial[iDecay]);
    chainMCInitial[iDecay]->SetBranchAddress("isDaughter", isDaughter_MCInitial[iDecay]);

    chainMCInitial[iDecay]->SetBranchAddress("p", p_MCInitial[iDecay]);
    chainMCInitial[iDecay]->SetBranchAddress("pT", pT_MCInitial[iDecay]);
    chainMCInitial[iDecay]->SetBranchAddress("eta", eta_MCInitial[iDecay]);
    chainMCInitial[iDecay]->SetBranchAddress("phi", phi_MCInitial[iDecay]);
    chainMCInitial[iDecay]->SetBranchAddress("mass", mass_MCInitial[iDecay]);
    chainMCInitial[iDecay]->SetBranchAddress("decayLength", decayLength_MCInitial[iDecay]);
    chainMCInitial[iDecay]->SetBranchAddress("decayLengthXY", decayLengthXY_MCInitial[iDecay]);
    chainMCInitial[iDecay]->SetBranchAddress("lifeTime", lifeTime_MCInitial[iDecay]);
    chainMCInitial[iDecay]->SetBranchAddress("r", r_MCInitial[iDecay]);
    chainMCInitial[iDecay]->SetBranchAddress("rapidity", rapidity_MCInitial[iDecay]);
    chainMCInitial[iDecay]->SetBranchAddress("theta", theta_MCInitial[iDecay]);
  }

  //--------------------------------------------------------------
  //Read MC Final tree
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

  TChain* chainMCFinal[NUM_OF_DECAYS];

  for (int iDecay = 0; iDecay < NUM_OF_DECAYS; iDecay++) {
    chainMCFinal[iDecay] = new TChain(nameOfMCFinalOutputTree[iDecay]);

    chainMCFinal[iDecay]->Add(KFInputFile);

    chainMCFinal[iDecay]->SetBranchAddress("nTracks", &nTracks_MCFinal[iDecay]);
    chainMCFinal[iDecay]->SetBranchAddress("numOfMCEvent", &numOfMCEvent_MCFinal[iDecay]);

    chainMCFinal[iDecay]->SetBranchAddress("ID", id_MCFinal[iDecay]);
    chainMCFinal[iDecay]->SetBranchAddress("ParentID", parentid_MCFinal[iDecay]);
    chainMCFinal[iDecay]->SetBranchAddress("pdgID", pdg_MCFinal[iDecay]);
    chainMCFinal[iDecay]->SetBranchAddress("x", x_MCFinal[iDecay]);
    chainMCFinal[iDecay]->SetBranchAddress("y", y_MCFinal[iDecay]);
    chainMCFinal[iDecay]->SetBranchAddress("z", z_MCFinal[iDecay]);
    chainMCFinal[iDecay]->SetBranchAddress("pX", pX_MCFinal[iDecay]);
    chainMCFinal[iDecay]->SetBranchAddress("pY", pY_MCFinal[iDecay]);
    chainMCFinal[iDecay]->SetBranchAddress("pZ", pZ_MCFinal[iDecay]);
    chainMCFinal[iDecay]->SetBranchAddress("e", e_MCFinal[iDecay]);
    chainMCFinal[iDecay]->SetBranchAddress("s", s_MCFinal[iDecay]);
    chainMCFinal[iDecay]->SetBranchAddress("charge", charge_MCFinal[iDecay]);

    chainMCFinal[iDecay]->SetBranchAddress("isMother", isMother_MCFinal[iDecay]);
    chainMCFinal[iDecay]->SetBranchAddress("isDaughter", isDaughter_MCFinal[iDecay]);

    chainMCFinal[iDecay]->SetBranchAddress("p", p_MCFinal[iDecay]);
    chainMCFinal[iDecay]->SetBranchAddress("pT", pT_MCFinal[iDecay]);
    chainMCFinal[iDecay]->SetBranchAddress("eta", eta_MCFinal[iDecay]);
    chainMCFinal[iDecay]->SetBranchAddress("phi", phi_MCFinal[iDecay]);
    chainMCFinal[iDecay]->SetBranchAddress("mass", mass_MCFinal[iDecay]);
    chainMCFinal[iDecay]->SetBranchAddress("decayLength", decayLength_MCFinal[iDecay]);
    chainMCFinal[iDecay]->SetBranchAddress("decayLengthXY", decayLengthXY_MCFinal[iDecay]);
    chainMCFinal[iDecay]->SetBranchAddress("lifeTime", lifeTime_MCFinal[iDecay]);
    chainMCFinal[iDecay]->SetBranchAddress("r", r_MCFinal[iDecay]);
    chainMCFinal[iDecay]->SetBranchAddress("rapidity", rapidity_MCFinal[iDecay]);
    chainMCFinal[iDecay]->SetBranchAddress("theta", theta_MCFinal[iDecay]);
  }

  //--------------------------------------------------------------
  //Read KFProductionVertex tree
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

  TChain* chainKFProductionVertex[NUM_OF_DECAYS];

  for (int iDecay = 0; iDecay < NUM_OF_DECAYS; iDecay++) {
    chainKFProductionVertex[iDecay] = new TChain(nameOfKFProductionVertexTree[iDecay]);

    chainKFProductionVertex[iDecay]->Add(KFInputFile);

    chainKFProductionVertex[iDecay]->SetBranchAddress("nTracks", &nTracks_KFProductionVertex[iDecay]);
    chainKFProductionVertex[iDecay]->SetBranchAddress("numOfMCEvent", &numOfMCEvent_KFProductionVertex[iDecay]);
    chainKFProductionVertex[iDecay]->SetBranchAddress("isAllCovMatOK", &isAllCovMatOK_KFProductionVertex[iDecay]);

    chainKFProductionVertex[iDecay]->SetBranchAddress("ID", id_KFProductionVertex[iDecay]);
    chainKFProductionVertex[iDecay]->SetBranchAddress("IDMC", idMC_KFProductionVertex[iDecay]);
    chainKFProductionVertex[iDecay]->SetBranchAddress("pdgID", pdg_KFProductionVertex[iDecay]);
    chainKFProductionVertex[iDecay]->SetBranchAddress("params", params_KFProductionVertex[iDecay]);
    chainKFProductionVertex[iDecay]->SetBranchAddress("covMat", covMat_KFProductionVertex[iDecay]);
    chainKFProductionVertex[iDecay]->SetBranchAddress("isCovMatOK", isCovMatOK_KFProductionVertex[iDecay]);
    chainKFProductionVertex[iDecay]->SetBranchAddress("x", x_KFProductionVertex[iDecay]);
    chainKFProductionVertex[iDecay]->SetBranchAddress("y", y_KFProductionVertex[iDecay]);
    chainKFProductionVertex[iDecay]->SetBranchAddress("z", z_KFProductionVertex[iDecay]);
    chainKFProductionVertex[iDecay]->SetBranchAddress("pX", pX_KFProductionVertex[iDecay]);
    chainKFProductionVertex[iDecay]->SetBranchAddress("pY", pY_KFProductionVertex[iDecay]);
    chainKFProductionVertex[iDecay]->SetBranchAddress("pZ", pZ_KFProductionVertex[iDecay]);
    chainKFProductionVertex[iDecay]->SetBranchAddress("e", e_KFProductionVertex[iDecay]);
    chainKFProductionVertex[iDecay]->SetBranchAddress("s", s_KFProductionVertex[iDecay]);
    chainKFProductionVertex[iDecay]->SetBranchAddress("charge", charge_KFProductionVertex[iDecay]);
    chainKFProductionVertex[iDecay]->SetBranchAddress("chi2", chi2_KFProductionVertex[iDecay]);
    chainKFProductionVertex[iDecay]->SetBranchAddress("ndf", ndf_KFProductionVertex[iDecay]);
    chainKFProductionVertex[iDecay]->SetBranchAddress("massHypo", massHypo_KFProductionVertex[iDecay]);

    chainKFProductionVertex[iDecay]->SetBranchAddress("isMother", isMother_KFProductionVertex[iDecay]);
    chainKFProductionVertex[iDecay]->SetBranchAddress("isDaughter", isDaughter_KFProductionVertex[iDecay]);

    chainKFProductionVertex[iDecay]->SetBranchAddress("p", p_KFProductionVertex[iDecay]);
    chainKFProductionVertex[iDecay]->SetBranchAddress("pT", pT_KFProductionVertex[iDecay]);
    chainKFProductionVertex[iDecay]->SetBranchAddress("eta", eta_KFProductionVertex[iDecay]);
    chainKFProductionVertex[iDecay]->SetBranchAddress("phi", phi_KFProductionVertex[iDecay]);
    chainKFProductionVertex[iDecay]->SetBranchAddress("mass", mass_KFProductionVertex[iDecay]);
    chainKFProductionVertex[iDecay]->SetBranchAddress("decayLength", decayLength_KFProductionVertex[iDecay]);
    chainKFProductionVertex[iDecay]->SetBranchAddress("decayLengthXY", decayLengthXY_KFProductionVertex[iDecay]);
    chainKFProductionVertex[iDecay]->SetBranchAddress("lifeTime", lifeTime_KFProductionVertex[iDecay]);
    chainKFProductionVertex[iDecay]->SetBranchAddress("r", r_KFProductionVertex[iDecay]);
    chainKFProductionVertex[iDecay]->SetBranchAddress("rapidity", rapidity_KFProductionVertex[iDecay]);
    chainKFProductionVertex[iDecay]->SetBranchAddress("theta", theta_KFProductionVertex[iDecay]);

    chainKFProductionVertex[iDecay]->SetBranchAddress("xErr", xErr_KFProductionVertex[iDecay]);
    chainKFProductionVertex[iDecay]->SetBranchAddress("yErr", yErr_KFProductionVertex[iDecay]);
    chainKFProductionVertex[iDecay]->SetBranchAddress("zErr", zErr_KFProductionVertex[iDecay]);
    chainKFProductionVertex[iDecay]->SetBranchAddress("pXErr", pXErr_KFProductionVertex[iDecay]);
    chainKFProductionVertex[iDecay]->SetBranchAddress("pYErr", pYErr_KFProductionVertex[iDecay]);
    chainKFProductionVertex[iDecay]->SetBranchAddress("pZErr", pZErr_KFProductionVertex[iDecay]);
    chainKFProductionVertex[iDecay]->SetBranchAddress("eErr", eErr_KFProductionVertex[iDecay]);
    chainKFProductionVertex[iDecay]->SetBranchAddress("sErr", sErr_KFProductionVertex[iDecay]);
    chainKFProductionVertex[iDecay]->SetBranchAddress("pErr", pErr_KFProductionVertex[iDecay]);
    chainKFProductionVertex[iDecay]->SetBranchAddress("pTErr", pTErr_KFProductionVertex[iDecay]);
    chainKFProductionVertex[iDecay]->SetBranchAddress("etaErr", etaErr_KFProductionVertex);
    chainKFProductionVertex[iDecay]->SetBranchAddress("phiErr", phiErr_KFProductionVertex[iDecay]);
    chainKFProductionVertex[iDecay]->SetBranchAddress("massErr", massErr_KFProductionVertex[iDecay]);
    chainKFProductionVertex[iDecay]->SetBranchAddress("decayLengthErr", decayLengthErr_KFProductionVertex);
    chainKFProductionVertex[iDecay]->SetBranchAddress("decayLengthXYErr", decayLengthXYErr_KFProductionVertex[iDecay]);
    chainKFProductionVertex[iDecay]->SetBranchAddress("lifeTimeErr", lifeTimeErr_KFProductionVertex[iDecay]);
    chainKFProductionVertex[iDecay]->SetBranchAddress("rErr", rErr_KFProductionVertex[iDecay]);
  }

  //--------------------------------------------------------------
  //Read KFDecayVertex tree
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

  TChain* chainKFDecayVertex[NUM_OF_DECAYS];

  for (int iDecay = 0; iDecay < NUM_OF_DECAYS; iDecay++) {
    chainKFDecayVertex[iDecay] = new TChain(nameOfKFDecayVertexTree[iDecay]);

    std::cout << "KFInputFile: " << KFInputFile << std::endl;
    chainKFDecayVertex[iDecay]->Add(KFInputFile);

    chainKFDecayVertex[iDecay]->SetBranchAddress("nTracks", &nTracks_KFDecayVertex[iDecay]);
    chainKFDecayVertex[iDecay]->SetBranchAddress("numOfMCEvent", &numOfMCEvent_KFDecayVertex[iDecay]);
    chainKFDecayVertex[iDecay]->SetBranchAddress("isAllCovMatOK", &isAllCovMatOK_KFDecayVertex[iDecay]);

    chainKFDecayVertex[iDecay]->SetBranchAddress("ID", id_KFDecayVertex[iDecay]);
    chainKFDecayVertex[iDecay]->SetBranchAddress("IDMC", idMC_KFDecayVertex[iDecay]);
    chainKFDecayVertex[iDecay]->SetBranchAddress("pdgID", pdg_KFDecayVertex[iDecay]);
    chainKFDecayVertex[iDecay]->SetBranchAddress("params", params_KFDecayVertex[iDecay]);
    chainKFDecayVertex[iDecay]->SetBranchAddress("covMat", covMat_KFDecayVertex[iDecay]);
    chainKFDecayVertex[iDecay]->SetBranchAddress("isCovMatOK", isCovMatOK_KFDecayVertex[iDecay]);
    chainKFDecayVertex[iDecay]->SetBranchAddress("x", x_KFDecayVertex[iDecay]);
    chainKFDecayVertex[iDecay]->SetBranchAddress("y", y_KFDecayVertex[iDecay]);
    chainKFDecayVertex[iDecay]->SetBranchAddress("z", z_KFDecayVertex[iDecay]);
    chainKFDecayVertex[iDecay]->SetBranchAddress("pX", pX_KFDecayVertex[iDecay]);
    chainKFDecayVertex[iDecay]->SetBranchAddress("pY", pY_KFDecayVertex[iDecay]);
    chainKFDecayVertex[iDecay]->SetBranchAddress("pZ", pZ_KFDecayVertex[iDecay]);
    chainKFDecayVertex[iDecay]->SetBranchAddress("e", e_KFDecayVertex[iDecay]);
    chainKFDecayVertex[iDecay]->SetBranchAddress("s", s_KFDecayVertex[iDecay]);
    chainKFDecayVertex[iDecay]->SetBranchAddress("charge", charge_KFDecayVertex[iDecay]);
    chainKFDecayVertex[iDecay]->SetBranchAddress("chi2", chi2_KFDecayVertex[iDecay]);
    chainKFDecayVertex[iDecay]->SetBranchAddress("ndf", ndf_KFDecayVertex[iDecay]);
    chainKFDecayVertex[iDecay]->SetBranchAddress("massHypo", massHypo_KFDecayVertex[iDecay]);

    chainKFDecayVertex[iDecay]->SetBranchAddress("isMother", isMother_KFDecayVertex[iDecay]);
    chainKFDecayVertex[iDecay]->SetBranchAddress("isDaughter", isDaughter_KFDecayVertex[iDecay]);

    chainKFDecayVertex[iDecay]->SetBranchAddress("p", p_KFDecayVertex[iDecay]);
    chainKFDecayVertex[iDecay]->SetBranchAddress("pT", pT_KFDecayVertex[iDecay]);
    chainKFDecayVertex[iDecay]->SetBranchAddress("eta", eta_KFDecayVertex[iDecay]);
    chainKFDecayVertex[iDecay]->SetBranchAddress("phi", phi_KFDecayVertex[iDecay]);
    chainKFDecayVertex[iDecay]->SetBranchAddress("mass", mass_KFDecayVertex[iDecay]);
    chainKFDecayVertex[iDecay]->SetBranchAddress("decayLength", decayLength_KFDecayVertex[iDecay]);
    chainKFDecayVertex[iDecay]->SetBranchAddress("decayLengthXY", decayLengthXY_KFDecayVertex[iDecay]);
    chainKFDecayVertex[iDecay]->SetBranchAddress("lifeTime", lifeTime_KFDecayVertex[iDecay]);
    chainKFDecayVertex[iDecay]->SetBranchAddress("r", r_KFDecayVertex[iDecay]);
    chainKFDecayVertex[iDecay]->SetBranchAddress("rapidity", rapidity_KFDecayVertex[iDecay]);
    chainKFDecayVertex[iDecay]->SetBranchAddress("theta", theta_KFDecayVertex[iDecay]);

    chainKFDecayVertex[iDecay]->SetBranchAddress("xErr", xErr_KFDecayVertex[iDecay]);
    chainKFDecayVertex[iDecay]->SetBranchAddress("yErr", yErr_KFDecayVertex[iDecay]);
    chainKFDecayVertex[iDecay]->SetBranchAddress("zErr", zErr_KFDecayVertex[iDecay]);
    chainKFDecayVertex[iDecay]->SetBranchAddress("pXErr", pXErr_KFDecayVertex[iDecay]);
    chainKFDecayVertex[iDecay]->SetBranchAddress("pYErr", pYErr_KFDecayVertex[iDecay]);
    chainKFDecayVertex[iDecay]->SetBranchAddress("pZErr", pZErr_KFDecayVertex[iDecay]);
    chainKFDecayVertex[iDecay]->SetBranchAddress("eErr", eErr_KFDecayVertex[iDecay]);
    chainKFDecayVertex[iDecay]->SetBranchAddress("sErr", sErr_KFDecayVertex[iDecay]);
    chainKFDecayVertex[iDecay]->SetBranchAddress("pErr", pErr_KFDecayVertex[iDecay]);
    chainKFDecayVertex[iDecay]->SetBranchAddress("pTErr", pTErr_KFDecayVertex[iDecay]);
    chainKFDecayVertex[iDecay]->SetBranchAddress("etaErr", etaErr_KFDecayVertex);
    chainKFDecayVertex[iDecay]->SetBranchAddress("phiErr", phiErr_KFDecayVertex[iDecay]);
    chainKFDecayVertex[iDecay]->SetBranchAddress("massErr", massErr_KFDecayVertex[iDecay]);
    chainKFDecayVertex[iDecay]->SetBranchAddress("decayLengthErr", decayLengthErr_KFDecayVertex);
    chainKFDecayVertex[iDecay]->SetBranchAddress("decayLengthXYErr", decayLengthXYErr_KFDecayVertex[iDecay]);
    chainKFDecayVertex[iDecay]->SetBranchAddress("lifeTimeErr", lifeTimeErr_KFDecayVertex[iDecay]);
    chainKFDecayVertex[iDecay]->SetBranchAddress("rErr", rErr_KFDecayVertex[iDecay]);
  }
  ////////////////////////////////////////////////////////////////////////
  ////////////////// end of trees block
  ////////////////////////////////////////////////////////////////////////

  ////////////////////////////////////////////////////////////////////////
  ////////////////// START of event loop block
  ////////////////////////////////////////////////////////////////////////

  cout << " Starting event loops\n";

  int nEventsMC = chainMC->GetEntries();
  int nEventsMCInitial[NUM_OF_DECAYS];
  int nEventsMCFinal[NUM_OF_DECAYS];
  int nEventsKFProductionVertex[NUM_OF_DECAYS];
  int nEventsKFDecayVertex[NUM_OF_DECAYS];
  for (int iDecay = 0; iDecay < NUM_OF_DECAYS; iDecay++) {
    nEventsMCInitial[iDecay]          = chainMCInitial[iDecay]->GetEntries();
    nEventsMCFinal[iDecay]            = chainMCFinal[iDecay]->GetEntries();
    nEventsKFProductionVertex[iDecay] = chainKFProductionVertex[iDecay]->GetEntries();
    nEventsKFDecayVertex[iDecay]      = chainKFDecayVertex[iDecay]->GetEntries();
    cout << "nEventsMCInitial[" << MOTHER_PDG_IDS[iDecay] << "] = " << nEventsMCInitial[iDecay] << endl;
    cout << "nEventsMCFinal[" << MOTHER_PDG_IDS[iDecay] << "] = " << nEventsMCFinal[iDecay] << endl;
    cout << "nEventsKFProductionVertex[" << MOTHER_PDG_IDS[iDecay] << "] = " << nEventsKFProductionVertex[iDecay]
         << endl;
    cout << "nEventsKFDecayVertex[" << MOTHER_PDG_IDS[iDecay] << "] = " << nEventsKFDecayVertex[iDecay] << endl;

    if (nEventsKFProductionVertex[iDecay] != nEventsKFDecayVertex[iDecay]) {
      cout << "nEventsKFProductionVertex != nEventsKFDecayVertex (" << nEventsKFProductionVertex[iDecay]
           << " != " << nEventsKFDecayVertex[iDecay] << ")\n. Quit.";
      return;
    }
    if (nEventsMCInitial[iDecay] != nEventsMCFinal[iDecay]) {
      cout << "nEventsMCInitial != nEventsMCFinal (" << nEventsMCInitial[iDecay] << " != " << nEventsMCFinal[iDecay]
           << ")\n. Quit.";
      return;
    }
    if (nEventsMCInitial[iDecay] != nEventsKFProductionVertex[iDecay]) {
      cout << "nEventsMCInitial != nEventsKFProductionVertex (" << nEventsMCInitial[iDecay]
           << " != " << nEventsKFProductionVertex[iDecay] << ")\n. Quit.";
      return;
    }
  }

  //int nEvents = nEventsKFProductionVertex;

  /* //OLD version MC QA plots from MC input tree, switching to QA from MCInitial and MCFinal trees
    // Create MC QA plots
    for (int iEvent = 0; iEvent < nEventsMC; iEvent++){
        chainMC.GetEntry(iEvent);
        // if number of tracks in the event less than
        // number of daughters + number of mother particles - do not process the event
        if (nTracks_MC < NUM_OF_KFPARTICLES)
            continue;
        
        if(iEvent % 1000 == 0) {
            std::cout << "processing " << iEvent << " MC event\r";
            std::cout << std::flush;
        }    

        size_t motherCounter = 0;
        size_t daughterCounter = 0;

        // loop over the tracks
        for (int iTrack = 0; iTrack < nTracks_MC; iTrack++){
            // if we found all daughters and mother - do not process other tracks
            if (daughterCounter >= NUM_OF_DAUGHTERS && motherCounter >= NUM_OF_MOTHERS)
                break;
            //TODO
            if (parentID_MC[iTrack]==0||parentID_MC[iTrack]==1){
                //setup mother particle from MC data and fill some histos
                cout << pdg_MC[iTrack] << " " << parentID_MC[iTrack] << " " << iTrack << endl;
                cout << MOTHER_PDG << " " <<  " MOTHER_PDG " << iTrack << endl;

                if (pdg_MC[iTrack] != MOTHER_PDG) 
                {
                    cout << "Mother particle's PDGID is not in the set of mothers pdgs... error and skip the event # " << iEvent << endl;
                    continue;
                }
                cout << "found it" << iEvent << endl;
                fHistXMotherMC -> Fill(finalX_MC[iTrack]);
                fHistYMotherMC -> Fill(finalY_MC[iTrack]);
                fHistZMotherMC -> Fill(finalZ_MC[iTrack]);
                fHistPxMotherMC -> Fill(finalPX_MC[iTrack]);
                fHistPyMotherMC -> Fill(finalPY_MC[iTrack]);
                fHistPzMotherMC -> Fill(finalPZ_MC[iTrack]);
                fHistChargeMotherMC -> Fill(charge_MC[iTrack]);
                fHistPDGMotherMC -> Fill(pdg_MC[iTrack]);
                fHistPtMotherMC -> Fill(pT(finalPX_MC[iTrack],finalPY_MC[iTrack]));
                fHistOneOverPtMotherMC -> Fill(1./pT(finalPX_MC[iTrack],finalPY_MC[iTrack]));
                fHistPMotherMC -> Fill(p(finalPX_MC[iTrack],finalPY_MC[iTrack], finalPZ_MC[iTrack]));
                fHistEMotherMC -> Fill(e(finalPX_MC[iTrack],finalPY_MC[iTrack],finalPZ_MC[iTrack],mass_MC[iTrack]));
                fHistMassMotherMC -> Fill(mass_MC[iTrack]);
                fHistEtaMotherMC -> Fill(eta(finalPX_MC[iTrack],finalPY_MC[iTrack], finalPZ_MC[iTrack]));
                fHistPhiMotherMC -> Fill(phi(finalPX_MC[iTrack],finalPY_MC[iTrack]));
                //fHistChi2MotherMC -> Fill();

                motherCounter++; 
            }
            if (parentID_MC[iTrack] == 1){
                auto iter_pdg = find (DAUGTHERS_PDGS.begin(), DAUGTHERS_PDGS.end(), pdg_MC[iTrack]);
                if ( iter_pdg == DAUGTHERS_PDGS.end()){
                    cout << "Daughter particle's PDGID is not in the set of daughters pdgs... error and skip the event # " << iEvent << endl;
                    break;
                }
                int daughterNum = distance(DAUGTHERS_PDGS.begin(),iter_pdg);
                // setup daughter particles from MC
                fHistXDaughterMC[daughterNum] -> Fill(initialX_MC[iTrack]);
                fHistYDaughterMC[daughterNum] -> Fill(initialY_MC[iTrack]);
                fHistZDaughterMC[daughterNum] -> Fill(initialZ_MC[iTrack]);
                fHistPxDaughterMC[daughterNum] -> Fill(initialPX_MC[iTrack]);
                fHistPyDaughterMC[daughterNum] -> Fill(initialPY_MC[iTrack]);
                fHistPzDaughterMC[daughterNum] -> Fill(initialPZ_MC[iTrack]);
                fHistChargeDaughterMC[daughterNum] -> Fill(charge_MC[iTrack]);
                fHistPDGDaughterMC[daughterNum] -> Fill(pdg_MC[iTrack]);
                fHistPtDaughterMC[daughterNum] -> Fill(pT(initialPX_MC[iTrack],initialPY_MC[iTrack]));
                fHistOneOverPtDaughterMC[daughterNum] -> Fill(1./pT(initialPX_MC[iTrack],initialPY_MC[iTrack]));
                fHistPDaughterMC[daughterNum] -> Fill(p(initialPX_MC[iTrack],initialPY_MC[iTrack], initialPZ_MC[iTrack]));
                fHistEDaughterMC[daughterNum] -> Fill(e(initialPX_MC[iTrack],initialPY_MC[iTrack],initialPZ_MC[iTrack],mass_MC[iTrack]));
                fHistMassDaughterMC[daughterNum] -> Fill(mass_MC[iTrack]);
                fHistEtaDaughterMC[daughterNum] -> Fill(eta(initialPX_MC[iTrack],initialPY_MC[iTrack], initialPZ_MC[iTrack]));
                fHistPhiDaughterMC[daughterNum] -> Fill(phi(initialPX_MC[iTrack],initialPY_MC[iTrack]));

                daughterCounter++;            
            }
        } // end of track loop

    } // end of MC event loop
    cout << endl;
    */

  // Create MCInitial QA plots
  for (int iDecay = 0; iDecay < NUM_OF_DECAYS; iDecay++) {
    std::cout << "processing decay: " << MOTHER_PDG_IDS[iDecay] << " MCInitial" << std::endl;
    for (int iEvent = 0; iEvent < nEventsMCInitial[iDecay]; iEvent++) {
      chainMCInitial[iDecay]->GetEntry(iEvent);
      // if number of tracks in the event less than
      // number of daughters + number of mother particles - do not process the event
      if (nTracks_MCInitial[iDecay] < NUM_OF_DAUGHTERS[iDecay] + 1) { continue; }
      if (iEvent % 1000 == 0) {
        std::cout << "processing " << iEvent << " MCInitial event\r";
        std::cout << std::flush;
      }

      //setup mother particle from MCInitial data and fill some histos
      fHistXMotherMCInitial[iDecay]->Fill(x_MCInitial[iDecay][0]);
      fHistYMotherMCInitial[iDecay]->Fill(y_MCInitial[iDecay][0]);
      fHistZMotherMCInitial[iDecay]->Fill(z_MCInitial[iDecay][0]);
      fHistPxMotherMCInitial[iDecay]->Fill(pX_MCInitial[iDecay][0]);
      fHistPyMotherMCInitial[iDecay]->Fill(pY_MCInitial[iDecay][0]);
      fHistPzMotherMCInitial[iDecay]->Fill(pZ_MCInitial[iDecay][0]);
      fHistChargeMotherMCInitial[iDecay]->Fill(charge_MCInitial[iDecay][0]);
      fHistPDGMotherMCInitial[iDecay]->Fill(pdg_MCInitial[iDecay][0]);
      fHistPtMotherMCInitial[iDecay]->Fill(pT(pX_MCInitial[iDecay][0], pY_MCInitial[iDecay][0]));
      fHistOneOverPtMotherMCInitial[iDecay]->Fill(1. / pT(pX_MCInitial[iDecay][0], pY_MCInitial[iDecay][0]));
      fHistPMotherMCInitial[iDecay]->Fill(p(pX_MCInitial[iDecay][0], pY_MCInitial[iDecay][0], pZ_MCInitial[iDecay][0]));
      fHistEMotherMCInitial[iDecay]->Fill(
        e(pX_MCInitial[iDecay][0], pY_MCInitial[iDecay][0], pZ_MCInitial[iDecay][0], mass_MCInitial[iDecay][0]));
      fHistMassMotherMCInitial[iDecay]->Fill(mass_MCInitial[iDecay][0]);
      fHistEtaMotherMCInitial[iDecay]->Fill(
        eta(pX_MCInitial[iDecay][0], pY_MCInitial[iDecay][0], pZ_MCInitial[iDecay][0]));
      fHistPhiMotherMCInitial[iDecay]->Fill(phi(pX_MCInitial[iDecay][0], pY_MCInitial[iDecay][0]));

      // loop over daughter tracks
      for (int iTrack = 1; iTrack < nTracks_MCInitial[iDecay]; iTrack++) {
        auto iter_pdg =
          find(DAUGHTER_PDG_IDS[iDecay].begin(), DAUGHTER_PDG_IDS[iDecay].end(), pdg_MCInitial[iDecay][iTrack]);
        if (iter_pdg == DAUGHTER_PDG_IDS[iDecay].end()) {
          cout << "MC Initial: Daughter particle's PDGID " << pdg_MCInitial[iDecay][iTrack]
               << " is not in the set of daughters pdgs... error and skip the event # " << iEvent << endl;
          break;
        }
        int daughterNum = distance(DAUGHTER_PDG_IDS[iDecay].begin(), iter_pdg);
        // setup daughter particles from MCInitial
        fHistXDaughterMCInitial[iDecay][daughterNum]->Fill(x_MCInitial[iDecay][iTrack]);
        fHistYDaughterMCInitial[iDecay][daughterNum]->Fill(y_MCInitial[iDecay][iTrack]);
        fHistZDaughterMCInitial[iDecay][daughterNum]->Fill(z_MCInitial[iDecay][iTrack]);
        fHistPxDaughterMCInitial[iDecay][daughterNum]->Fill(pX_MCInitial[iDecay][iTrack]);
        fHistPyDaughterMCInitial[iDecay][daughterNum]->Fill(pY_MCInitial[iDecay][iTrack]);
        fHistPzDaughterMCInitial[iDecay][daughterNum]->Fill(pZ_MCInitial[iDecay][iTrack]);
        fHistChargeDaughterMCInitial[iDecay][daughterNum]->Fill(charge_MCInitial[iDecay][iTrack]);
        fHistPDGDaughterMCInitial[iDecay][daughterNum]->Fill(pdg_MCInitial[iDecay][iTrack]);
        fHistPtDaughterMCInitial[iDecay][daughterNum]->Fill(
          pT(pX_MCInitial[iDecay][iTrack], pY_MCInitial[iDecay][iTrack]));
        fHistOneOverPtDaughterMCInitial[iDecay][daughterNum]->Fill(
          1. / pT(pX_MCInitial[iDecay][iTrack], pY_MCInitial[iDecay][iTrack]));
        fHistPDaughterMCInitial[iDecay][daughterNum]->Fill(
          p(pX_MCInitial[iDecay][iTrack], pY_MCInitial[iDecay][iTrack], pZ_MCInitial[iDecay][iTrack]));
        fHistEDaughterMCInitial[iDecay][daughterNum]->Fill(e(pX_MCInitial[iDecay][iTrack], pY_MCInitial[iDecay][iTrack],
                                                             pZ_MCInitial[iDecay][iTrack],
                                                             mass_MCInitial[iDecay][iTrack]));
        fHistMassDaughterMCInitial[iDecay][daughterNum]->Fill(mass_MCInitial[iDecay][iTrack]);
        fHistEtaDaughterMCInitial[iDecay][daughterNum]->Fill(
          eta(pX_MCInitial[iDecay][iTrack], pY_MCInitial[iDecay][iTrack], pZ_MCInitial[iDecay][iTrack]));
        fHistPhiDaughterMCInitial[iDecay][daughterNum]->Fill(
          phi(pX_MCInitial[iDecay][iTrack], pY_MCInitial[iDecay][iTrack]));
      }  // end of track loop
    }  // end of event loop
    cout << endl;
  }  // end of decay loop

  // Create MCFinal QA plots
  for (int iDecay = 0; iDecay < NUM_OF_DECAYS; iDecay++) {
    std::cout << "processing decay: " << MOTHER_PDG_IDS[iDecay] << " MCFinal" << std::endl;
    for (int iEvent = 0; iEvent < nEventsMCFinal[iDecay]; iEvent++) {
      chainMCFinal[iDecay]->GetEntry(iEvent);
      // if number of tracks in the event less than
      // number of daughters + number of mother particles - do not process the event
      if (nTracks_MCFinal[iDecay] < NUM_OF_DAUGHTERS[iDecay] + 1) { continue; }
      if (iEvent % 1000 == 0) {
        std::cout << "processing " << iEvent << " MCFinal event\r";
        std::cout << std::flush;
      }

      //setup mother particle from MCFinal data and fill some histos
      fHistXMotherMCFinal[iDecay]->Fill(x_MCFinal[iDecay][0]);
      fHistYMotherMCFinal[iDecay]->Fill(y_MCFinal[iDecay][0]);
      fHistZMotherMCFinal[iDecay]->Fill(z_MCFinal[iDecay][0]);
      fHistPxMotherMCFinal[iDecay]->Fill(pX_MCFinal[iDecay][0]);
      fHistPyMotherMCFinal[iDecay]->Fill(pY_MCFinal[iDecay][0]);
      fHistPzMotherMCFinal[iDecay]->Fill(pZ_MCFinal[iDecay][0]);
      fHistChargeMotherMCFinal[iDecay]->Fill(charge_MCFinal[iDecay][0]);
      fHistPDGMotherMCFinal[iDecay]->Fill(pdg_MCFinal[iDecay][0]);
      fHistPtMotherMCFinal[iDecay]->Fill(pT(pX_MCFinal[iDecay][0], pY_MCFinal[iDecay][0]));
      fHistOneOverPtMotherMCFinal[iDecay]->Fill(1. / pT(pX_MCFinal[iDecay][0], pY_MCFinal[iDecay][0]));
      fHistPMotherMCFinal[iDecay]->Fill(p(pX_MCFinal[iDecay][0], pY_MCFinal[iDecay][0], pZ_MCFinal[iDecay][0]));
      fHistEMotherMCFinal[iDecay]->Fill(
        e(pX_MCFinal[iDecay][0], pY_MCFinal[iDecay][0], pZ_MCFinal[iDecay][0], mass_MCFinal[iDecay][0]));
      fHistMassMotherMCFinal[iDecay]->Fill(mass_MCFinal[iDecay][0]);
      fHistEtaMotherMCFinal[iDecay]->Fill(eta(pX_MCFinal[iDecay][0], pY_MCFinal[iDecay][0], pZ_MCFinal[iDecay][0]));
      fHistPhiMotherMCFinal[iDecay]->Fill(phi(pX_MCFinal[iDecay][0], pY_MCFinal[iDecay][0]));

      // loop over daughter tracks
      for (int iTrack = 1; iTrack < nTracks_MCFinal[iDecay]; iTrack++) {
        auto iter_pdg =
          find(DAUGHTER_PDG_IDS[iDecay].begin(), DAUGHTER_PDG_IDS[iDecay].end(), pdg_MCFinal[iDecay][iTrack]);
        if (iter_pdg == DAUGHTER_PDG_IDS[iDecay].end()) {
          cout << "MC Final: Daughter particle's PDGID " << pdg_MCFinal[iDecay][iTrack]
               << " is not in the set of daughters pdgs... error and skip the event # " << iEvent << endl;
          break;
        }
        int daughterNum = distance(DAUGHTER_PDG_IDS[iDecay].begin(), iter_pdg);
        // setup daughter particles from MCFinal
        fHistXDaughterMCFinal[iDecay][daughterNum]->Fill(x_MCFinal[iDecay][iTrack]);
        fHistYDaughterMCFinal[iDecay][daughterNum]->Fill(y_MCFinal[iDecay][iTrack]);
        fHistZDaughterMCFinal[iDecay][daughterNum]->Fill(z_MCFinal[iDecay][iTrack]);
        fHistPxDaughterMCFinal[iDecay][daughterNum]->Fill(pX_MCFinal[iDecay][iTrack]);
        fHistPyDaughterMCFinal[iDecay][daughterNum]->Fill(pY_MCFinal[iDecay][iTrack]);
        fHistPzDaughterMCFinal[iDecay][daughterNum]->Fill(pZ_MCFinal[iDecay][iTrack]);
        fHistChargeDaughterMCFinal[iDecay][daughterNum]->Fill(charge_MCFinal[iDecay][iTrack]);
        fHistPDGDaughterMCFinal[iDecay][daughterNum]->Fill(pdg_MCFinal[iDecay][iTrack]);
        fHistPtDaughterMCFinal[iDecay][daughterNum]->Fill(pT(pX_MCFinal[iDecay][iTrack], pY_MCFinal[iDecay][iTrack]));
        fHistOneOverPtDaughterMCFinal[iDecay][daughterNum]->Fill(
          1. / pT(pX_MCFinal[iDecay][iTrack], pY_MCFinal[iDecay][iTrack]));
        fHistPDaughterMCFinal[iDecay][daughterNum]->Fill(
          p(pX_MCFinal[iDecay][iTrack], pY_MCFinal[iDecay][iTrack], pZ_MCFinal[iDecay][iTrack]));
        fHistEDaughterMCFinal[iDecay][daughterNum]->Fill(e(pX_MCFinal[iDecay][iTrack], pY_MCFinal[iDecay][iTrack],
                                                           pZ_MCFinal[iDecay][iTrack], mass_MCFinal[iDecay][iTrack]));
        fHistMassDaughterMCFinal[iDecay][daughterNum]->Fill(mass_MCFinal[iDecay][iTrack]);
        fHistEtaDaughterMCFinal[iDecay][daughterNum]->Fill(
          eta(pX_MCFinal[iDecay][iTrack], pY_MCFinal[iDecay][iTrack], pZ_MCFinal[iDecay][iTrack]));
        fHistPhiDaughterMCFinal[iDecay][daughterNum]->Fill(phi(pX_MCFinal[iDecay][iTrack], pY_MCFinal[iDecay][iTrack]));
      }  // end of track loop
    }  // end of event loop
    cout << endl;
  }  // end of decay loop

  // Create KFProductionVertex QA plots
  for (int iDecay = 0; iDecay < NUM_OF_DECAYS; iDecay++) {
    std::cout << "processing decay: " << MOTHER_PDG_IDS[iDecay] << " KFProductionVertex" << std::endl;
    for (int iEvent = 0; iEvent < nEventsKFProductionVertex[iDecay]; iEvent++) {
      chainKFProductionVertex[iDecay]->GetEntry(iEvent);
      // if number of tracks in the event less than
      // number of daughters + number of mother particles - do not process the event
      if (nTracks_KFProductionVertex[iDecay] < NUM_OF_DAUGHTERS[iDecay] + 1) { continue; }
      if (!isAllCovMatOK_KFProductionVertex[iDecay]) { continue; }
      if (iEvent % 1000 == 0) {
        std::cout << "processing " << iEvent << " KFProductionVertex event\r";
        std::cout << std::flush;
      }

      //setup mother particle from KFProductionVertex data and fill some histos
      fHistXMotherKFProductionVertex[iDecay]->Fill(x_KFProductionVertex[iDecay][0]);
      fHistYMotherKFProductionVertex[iDecay]->Fill(y_KFProductionVertex[iDecay][0]);
      fHistZMotherKFProductionVertex[iDecay]->Fill(z_KFProductionVertex[iDecay][0]);
      fHistPxMotherKFProductionVertex[iDecay]->Fill(pX_KFProductionVertex[iDecay][0]);
      fHistPyMotherKFProductionVertex[iDecay]->Fill(pY_KFProductionVertex[iDecay][0]);
      fHistPzMotherKFProductionVertex[iDecay]->Fill(pZ_KFProductionVertex[iDecay][0]);
      fHistChargeMotherKFProductionVertex[iDecay]->Fill(charge_KFProductionVertex[iDecay][0]);
      fHistPDGMotherKFProductionVertex[iDecay]->Fill(pdg_KFProductionVertex[iDecay][0]);
      fHistPtMotherKFProductionVertex[iDecay]->Fill(
        pT(pX_KFProductionVertex[iDecay][0], pY_KFProductionVertex[iDecay][0]));
      fHistOneOverPtMotherKFProductionVertex[iDecay]->Fill(
        1. / pT(pX_KFProductionVertex[iDecay][0], pY_KFProductionVertex[iDecay][0]));
      fHistPMotherKFProductionVertex[iDecay]->Fill(
        p(pX_KFProductionVertex[iDecay][0], pY_KFProductionVertex[iDecay][0], pZ_KFProductionVertex[iDecay][0]));
      fHistEMotherKFProductionVertex[iDecay]->Fill(e(pX_KFProductionVertex[iDecay][0], pY_KFProductionVertex[iDecay][0],
                                                     pZ_KFProductionVertex[iDecay][0],
                                                     mass_KFProductionVertex[iDecay][0]));
      fHistMassMotherKFProductionVertex[iDecay]->Fill(mass_KFProductionVertex[iDecay][0]);
      fHistEtaMotherKFProductionVertex[iDecay]->Fill(
        eta(pX_KFProductionVertex[iDecay][0], pY_KFProductionVertex[iDecay][0], pZ_KFProductionVertex[iDecay][0]));
      fHistPhiMotherKFProductionVertex[iDecay]->Fill(
        phi(pX_KFProductionVertex[iDecay][0], pY_KFProductionVertex[iDecay][0]));
      fHistChi2MotherKFProductionVertex[iDecay]->Fill(chi2_KFProductionVertex[iDecay][0]);
      fHistNDFMotherKFProductionVertex[iDecay]->Fill(ndf_KFProductionVertex[iDecay][0]);

      // loop over daughter tracks
      for (int iTrack = 1; iTrack < nTracks_KFProductionVertex[iDecay]; iTrack++) {
        auto iter_pdg = find(DAUGHTER_PDG_IDS[iDecay].begin(), DAUGHTER_PDG_IDS[iDecay].end(),
                             pdg_KFProductionVertex[iDecay][iTrack]);
        if (iter_pdg == DAUGHTER_PDG_IDS[iDecay].end()) {
          cout << "KFProductionVertex: Daughter particle's PDGID " << pdg_KFProductionVertex[iDecay][iTrack]
               << " is not in the set of daughters pdgs... error and skip the event # " << iEvent << endl;
          break;
        }
        int daughterNum = distance(DAUGHTER_PDG_IDS[iDecay].begin(), iter_pdg);
        // setup daughter particles from KFProductionVertex
        fHistXDaughterKFProductionVertex[iDecay][daughterNum]->Fill(x_KFProductionVertex[iDecay][iTrack]);
        fHistYDaughterKFProductionVertex[iDecay][daughterNum]->Fill(y_KFProductionVertex[iDecay][iTrack]);
        fHistZDaughterKFProductionVertex[iDecay][daughterNum]->Fill(z_KFProductionVertex[iDecay][iTrack]);
        fHistPxDaughterKFProductionVertex[iDecay][daughterNum]->Fill(pX_KFProductionVertex[iDecay][iTrack]);
        fHistPyDaughterKFProductionVertex[iDecay][daughterNum]->Fill(pY_KFProductionVertex[iDecay][iTrack]);
        fHistPzDaughterKFProductionVertex[iDecay][daughterNum]->Fill(pZ_KFProductionVertex[iDecay][iTrack]);
        fHistChargeDaughterKFProductionVertex[iDecay][daughterNum]->Fill(charge_KFProductionVertex[iDecay][iTrack]);
        fHistPDGDaughterKFProductionVertex[iDecay][daughterNum]->Fill(pdg_KFProductionVertex[iDecay][iTrack]);
        fHistPtDaughterKFProductionVertex[iDecay][daughterNum]->Fill(
          pT(pX_KFProductionVertex[iDecay][iTrack], pY_KFProductionVertex[iDecay][iTrack]));
        fHistOneOverPtDaughterKFProductionVertex[iDecay][daughterNum]->Fill(
          1. / pT(pX_KFProductionVertex[iDecay][iTrack], pY_KFProductionVertex[iDecay][iTrack]));
        fHistPDaughterKFProductionVertex[iDecay][daughterNum]->Fill(p(pX_KFProductionVertex[iDecay][iTrack],
                                                                      pY_KFProductionVertex[iDecay][iTrack],
                                                                      pZ_KFProductionVertex[iDecay][iTrack]));
        fHistEDaughterKFProductionVertex[iDecay][daughterNum]->Fill(
          e(pX_KFProductionVertex[iDecay][iTrack], pY_KFProductionVertex[iDecay][iTrack],
            pZ_KFProductionVertex[iDecay][iTrack], mass_KFProductionVertex[iDecay][iTrack]));
        fHistMassDaughterKFProductionVertex[iDecay][daughterNum]->Fill(mass_KFProductionVertex[iDecay][iTrack]);
        fHistEtaDaughterKFProductionVertex[iDecay][daughterNum]->Fill(eta(pX_KFProductionVertex[iDecay][iTrack],
                                                                          pY_KFProductionVertex[iDecay][iTrack],
                                                                          pZ_KFProductionVertex[iDecay][iTrack]));
        fHistPhiDaughterKFProductionVertex[iDecay][daughterNum]->Fill(
          phi(pX_KFProductionVertex[iDecay][iTrack], pY_KFProductionVertex[iDecay][iTrack]));
        fHistChi2DaughterKFProductionVertex[iDecay][daughterNum]->Fill(chi2_KFProductionVertex[iDecay][iTrack]);
        fHistNDFDaughterKFProductionVertex[iDecay][daughterNum]->Fill(ndf_KFProductionVertex[iDecay][iTrack]);
      }  // end of track loop
    }  // end of event loop
    cout << endl;
  }  // end of decay loop

  //BEGIN TEST
  std::array<int, NUM_OF_DECAYS> event_counts;
  for (int i = 0; i < NUM_OF_DECAYS; i++) { event_counts[i] = 0; }
  //END TEST
  // Create KFDecayVertex QA plots
  for (int iDecay = 0; iDecay < NUM_OF_DECAYS; iDecay++) {
    std::cout << "processing decay: " << MOTHER_PDG_IDS[iDecay] << " KFDecayVertex" << std::endl;
    for (int iEvent = 0; iEvent < nEventsKFDecayVertex[iDecay]; iEvent++) {
      chainKFDecayVertex[iDecay]->GetEntry(iEvent);
      // if number of tracks in the event less than
      // number of daughters + number of mother particles - do not process the event
      // if (nTracks_KFDecayVertex[iDecay] < NUM_OF_DAUGHTERS[iDecay]+1)
      //     continue;
      if (!isAllCovMatOK_KFDecayVertex[iDecay]) { continue; }
      event_counts[iDecay]++;  //TEST
      if (iEvent % 1000 == 0) {
        std::cout << "processing " << iEvent << " KFDecayVertex event\r";
        std::cout << std::flush;
      }

      //setup mother particle from KFDecayVertex data and fill some histos
      fHistXMotherKFDecayVertex[iDecay]->Fill(x_KFDecayVertex[iDecay][0]);
      fHistYMotherKFDecayVertex[iDecay]->Fill(y_KFDecayVertex[iDecay][0]);
      fHistZMotherKFDecayVertex[iDecay]->Fill(z_KFDecayVertex[iDecay][0]);
      fHistPxMotherKFDecayVertex[iDecay]->Fill(pX_KFDecayVertex[iDecay][0]);
      fHistPyMotherKFDecayVertex[iDecay]->Fill(pY_KFDecayVertex[iDecay][0]);
      fHistPzMotherKFDecayVertex[iDecay]->Fill(pZ_KFDecayVertex[iDecay][0]);
      fHistChargeMotherKFDecayVertex[iDecay]->Fill(charge_KFDecayVertex[iDecay][0]);
      fHistPDGMotherKFDecayVertex[iDecay]->Fill(pdg_KFDecayVertex[iDecay][0]);
      fHistPtMotherKFDecayVertex[iDecay]->Fill(pT(pX_KFDecayVertex[iDecay][0], pY_KFDecayVertex[iDecay][0]));
      fHistOneOverPtMotherKFDecayVertex[iDecay]->Fill(1.
                                                      / pT(pX_KFDecayVertex[iDecay][0], pY_KFDecayVertex[iDecay][0]));
      fHistPMotherKFDecayVertex[iDecay]->Fill(
        p(pX_KFDecayVertex[iDecay][0], pY_KFDecayVertex[iDecay][0], pZ_KFDecayVertex[iDecay][0]));
      fHistEMotherKFDecayVertex[iDecay]->Fill(e(pX_KFDecayVertex[iDecay][0], pY_KFDecayVertex[iDecay][0],
                                                pZ_KFDecayVertex[iDecay][0], mass_KFDecayVertex[iDecay][0]));
      fHistMassMotherKFDecayVertex[iDecay]->Fill(mass_KFDecayVertex[iDecay][0]);
      fHistEtaMotherKFDecayVertex[iDecay]->Fill(
        eta(pX_KFDecayVertex[iDecay][0], pY_KFDecayVertex[iDecay][0], pZ_KFDecayVertex[iDecay][0]));
      fHistPhiMotherKFDecayVertex[iDecay]->Fill(phi(pX_KFDecayVertex[iDecay][0], pY_KFDecayVertex[iDecay][0]));
      fHistChi2MotherKFDecayVertex[iDecay]->Fill(chi2_KFDecayVertex[iDecay][0]);
      fHistNDFMotherKFDecayVertex[iDecay]->Fill(ndf_KFDecayVertex[iDecay][0]);

      // loop over daughter tracks
      for (int iTrack = 1; iTrack < nTracks_KFDecayVertex[iDecay]; iTrack++) {
        auto iter_pdg =
          find(DAUGHTER_PDG_IDS[iDecay].begin(), DAUGHTER_PDG_IDS[iDecay].end(), pdg_KFDecayVertex[iDecay][iTrack]);
        if (iter_pdg == DAUGHTER_PDG_IDS[iDecay].end()) {
          cout << "KFDecayVertex: Daughter particle's PDGID " << pdg_KFDecayVertex[iDecay][iTrack]
               << " is not in the set of daughters pdgs... error and skip the event # " << iEvent << endl;
          break;
        }
        int daughterNum = distance(DAUGHTER_PDG_IDS[iDecay].begin(), iter_pdg);
        // setup daughter particles from KFDecayVertex
        fHistXDaughterKFDecayVertex[iDecay][daughterNum]->Fill(x_KFDecayVertex[iDecay][iTrack]);
        fHistYDaughterKFDecayVertex[iDecay][daughterNum]->Fill(y_KFDecayVertex[iDecay][iTrack]);
        fHistZDaughterKFDecayVertex[iDecay][daughterNum]->Fill(z_KFDecayVertex[iDecay][iTrack]);
        fHistPxDaughterKFDecayVertex[iDecay][daughterNum]->Fill(pX_KFDecayVertex[iDecay][iTrack]);
        fHistPyDaughterKFDecayVertex[iDecay][daughterNum]->Fill(pY_KFDecayVertex[iDecay][iTrack]);
        fHistPzDaughterKFDecayVertex[iDecay][daughterNum]->Fill(pZ_KFDecayVertex[iDecay][iTrack]);
        fHistChargeDaughterKFDecayVertex[iDecay][daughterNum]->Fill(charge_KFDecayVertex[iDecay][iTrack]);
        fHistPDGDaughterKFDecayVertex[iDecay][daughterNum]->Fill(pdg_KFDecayVertex[iDecay][iTrack]);
        fHistPtDaughterKFDecayVertex[iDecay][daughterNum]->Fill(
          pT(pX_KFDecayVertex[iDecay][iTrack], pY_KFDecayVertex[iDecay][iTrack]));
        fHistOneOverPtDaughterKFDecayVertex[iDecay][daughterNum]->Fill(
          1. / pT(pX_KFDecayVertex[iDecay][iTrack], pY_KFDecayVertex[iDecay][iTrack]));
        fHistPDaughterKFDecayVertex[iDecay][daughterNum]->Fill(
          p(pX_KFDecayVertex[iDecay][iTrack], pY_KFDecayVertex[iDecay][iTrack], pZ_KFDecayVertex[iDecay][iTrack]));
        fHistEDaughterKFDecayVertex[iDecay][daughterNum]->Fill(
          e(pX_KFDecayVertex[iDecay][iTrack], pY_KFDecayVertex[iDecay][iTrack], pZ_KFDecayVertex[iDecay][iTrack],
            mass_KFDecayVertex[iDecay][iTrack]));
        fHistMassDaughterKFDecayVertex[iDecay][daughterNum]->Fill(mass_KFDecayVertex[iDecay][iTrack]);
        fHistEtaDaughterKFDecayVertex[iDecay][daughterNum]->Fill(
          eta(pX_KFDecayVertex[iDecay][iTrack], pY_KFDecayVertex[iDecay][iTrack], pZ_KFDecayVertex[iDecay][iTrack]));
        fHistPhiDaughterKFDecayVertex[iDecay][daughterNum]->Fill(
          phi(pX_KFDecayVertex[iDecay][iTrack], pY_KFDecayVertex[iDecay][iTrack]));
        fHistChi2DaughterKFDecayVertex[iDecay][daughterNum]->Fill(chi2_KFDecayVertex[iDecay][iTrack]);
        fHistNDFDaughterKFDecayVertex[iDecay][daughterNum]->Fill(ndf_KFDecayVertex[iDecay][iTrack]);
      }  // end of track loop
    }  // end of event loop
    cout << endl;
  }  // end of decay loop

  // Create Pt dep plots of residuals and pulls
  for (int iDecay = 0; iDecay < NUM_OF_DECAYS; iDecay++) {
    for (int iEvent = 0; iEvent < nEventsKFDecayVertex[iDecay]; iEvent++) {
      chainKFDecayVertex[iDecay]->GetEntry(iEvent);
      chainKFProductionVertex[iDecay]->GetEntry(iEvent);
      chainMCInitial[iDecay]->GetEntry(numOfMCEvent_KFDecayVertex[iDecay]);
      chainMCFinal[iDecay]->GetEntry(numOfMCEvent_KFDecayVertex[iDecay]);
      //chainMC.GetEntry(numOfMCEvent_KFDecayVertex);
      // if number of tracks in the event less than
      // number of daughters + number of mother particles - do not process the event
      if (!isAllCovMatOK_KFDecayVertex[iDecay]) { continue; }
      if (nTracks_KFDecayVertex[iDecay] < NUM_OF_DAUGHTERS[iDecay] + 1) { continue; }

      //size_t motherCounter = 0;
      //size_t daughterCounter = 0;

      // loop over the tracks
      for (int iTrack = 0; iTrack < nTracks_KFDecayVertex[iDecay]; iTrack++) {
        if (isMother_KFDecayVertex[iDecay][iTrack]) {
          //setup mother particle from KFDecayVertex data and fill some histos
          if (pdg_KFDecayVertex[iDecay][iTrack] != MOTHER_PDG_IDS[iDecay]) {
            cout << "Mother particle's PDGID " << pdg_KFDecayVertex[iDecay][iTrack]
                 << " is not equal to the mother pdg(" << MOTHER_PDG_IDS[iDecay] << ")... error and skip the event # "
                 << iEvent << endl;
            break;
          }
        }
        if (isDaughter_KFDecayVertex[iDecay][iTrack]) {
          auto iter_pdg =
            find(DAUGHTER_PDG_IDS[iDecay].begin(), DAUGHTER_PDG_IDS[iDecay].end(), pdg_KFDecayVertex[iDecay][iTrack]);
          if (iter_pdg == DAUGHTER_PDG_IDS[iDecay].end()) {
            cout << "KFDecayVertex: Daughter particle's PDGID " << pdg_KFDecayVertex[iDecay][iTrack]
                 << " is not in the set of daughters pdgs... error and skip the event # " << iEvent << endl;
            break;
          }
        }
        //int daughterNum = distance(DAUGHTER_PDG_IDS[iDecay].begin(),iter_pdg);

        MCParticleStruct mcpart_;
        KFParticleStruct kfpart_decay;
        KFParticleStruct kfpart_production;

        int iTrack_MC = iTrack;  //TEST idMC_KFDecayVertex[iDecay][iTrack];

        mcpart_.trackID     = id_MCInitial[iDecay][iTrack_MC];
        mcpart_.parentID    = parentid_MCInitial[iDecay][iTrack_MC];
        mcpart_.pdg         = pdg_MCInitial[iDecay][iTrack_MC];
        mcpart_.charge      = charge_MCInitial[iDecay][iTrack_MC];
        mcpart_.mass        = mass_MCInitial[iDecay][iTrack_MC];
        mcpart_.isMother    = isMother_KFDecayVertex[iDecay][iTrack];
        mcpart_.trackLength = decayLength_MCInitial[iDecay][iTrack_MC];

        mcpart_.initialX = x_MCInitial[iDecay][iTrack_MC];
        mcpart_.initialY = y_MCInitial[iDecay][iTrack_MC];
        mcpart_.initialZ = z_MCInitial[iDecay][iTrack_MC];

        mcpart_.finalX = x_MCFinal[iDecay][iTrack_MC];
        mcpart_.finalY = y_MCFinal[iDecay][iTrack_MC];
        mcpart_.finalZ = z_MCFinal[iDecay][iTrack_MC];

        mcpart_.initialPX = pX_MCInitial[iDecay][iTrack_MC];
        mcpart_.initialPY = pY_MCInitial[iDecay][iTrack_MC];
        mcpart_.initialPZ = pZ_MCInitial[iDecay][iTrack_MC];

        mcpart_.finalPX = pX_MCFinal[iDecay][iTrack_MC];
        mcpart_.finalPY = pY_MCFinal[iDecay][iTrack_MC];
        mcpart_.finalPZ = pZ_MCFinal[iDecay][iTrack_MC];
        ///mcpart_.ndf = 0.;
        ///mcpart_.chi2 = 0.;

        kfpart_decay.id   = iTrack;
        kfpart_decay.idMC = mcpart_.trackID;
        kfpart_decay.pdg  = pdg_KFDecayVertex[iDecay][iTrack];
        memcpy(kfpart_decay.params, params_KFDecayVertex[iDecay], sizeof(kfpart_decay.params));
        memcpy(kfpart_decay.covMat, covMat_KFDecayVertex[iDecay], sizeof(kfpart_decay.covMat));
        kfpart_decay.isCovMatOK = isCovMatOK_KFDecayVertex[iDecay][iTrack];
        kfpart_decay.x          = x_KFDecayVertex[iDecay][iTrack];
        kfpart_decay.y          = y_KFDecayVertex[iDecay][iTrack];
        kfpart_decay.z          = z_KFDecayVertex[iDecay][iTrack];
        kfpart_decay.pX         = pX_KFDecayVertex[iDecay][iTrack];
        kfpart_decay.pY         = pY_KFDecayVertex[iDecay][iTrack];
        kfpart_decay.pZ         = pZ_KFDecayVertex[iDecay][iTrack];
        kfpart_decay.e          = e_KFDecayVertex[iDecay][iTrack];
        kfpart_decay.s          = s_KFDecayVertex[iDecay][iTrack];
        kfpart_decay.charge     = charge_KFDecayVertex[iDecay][iTrack];
        kfpart_decay.chi2       = chi2_KFDecayVertex[iDecay][iTrack];
        kfpart_decay.ndf = ndf_KFDecayVertex[iDecay][iTrack];
        kfpart_decay.massHypo = massHypo_KFDecayVertex[iDecay][iTrack];

        kfpart_decay.isMother   = isMother_KFDecayVertex[iDecay][iTrack];
        kfpart_decay.isDaughter = isDaughter_KFDecayVertex[iDecay][iTrack];

        kfpart_decay.xErr          = xErr_KFDecayVertex[iDecay][iTrack];
        kfpart_decay.yErr          = yErr_KFDecayVertex[iDecay][iTrack];
        kfpart_decay.zErr          = zErr_KFDecayVertex[iDecay][iTrack];
        kfpart_decay.pXErr         = pXErr_KFDecayVertex[iDecay][iTrack];
        kfpart_decay.pYErr         = pYErr_KFDecayVertex[iDecay][iTrack];
        kfpart_decay.pZErr         = pZErr_KFDecayVertex[iDecay][iTrack];
        kfpart_decay.p             = pErr_KFDecayVertex[iDecay][iTrack];
        kfpart_decay.pT            = pT_KFDecayVertex[iDecay][iTrack];
        kfpart_decay.eta           = eta_KFDecayVertex[iDecay][iTrack];
        kfpart_decay.phi           = phi_KFDecayVertex[iDecay][iTrack];
        kfpart_decay.mass          = mass_KFDecayVertex[iDecay][iTrack];
        kfpart_decay.decayLength   = decayLength_KFDecayVertex[iDecay][iTrack];
        kfpart_decay.decayLengthXY = decayLengthXY_KFDecayVertex[iDecay][iTrack];
        kfpart_decay.lifeTime      = lifeTime_KFDecayVertex[iDecay][iTrack];
        kfpart_decay.r             = r_KFDecayVertex[iDecay][iTrack];
        kfpart_decay.rapidity      = rapidity_KFDecayVertex[iDecay][iTrack];
        kfpart_decay.theta         = theta_KFDecayVertex[iDecay][iTrack];

        kfpart_decay.eErr             = eErr_KFDecayVertex[iDecay][iTrack];
        kfpart_decay.sErr             = sErr_KFDecayVertex[iDecay][iTrack];
        kfpart_decay.pErr             = pErr_KFDecayVertex[iDecay][iTrack];
        kfpart_decay.pTErr            = pTErr_KFDecayVertex[iDecay][iTrack];
        kfpart_decay.etaErr           = etaErr_KFDecayVertex[iDecay][iTrack];
        kfpart_decay.phiErr           = phiErr_KFDecayVertex[iDecay][iTrack];
        kfpart_decay.massErr          = massErr_KFDecayVertex[iDecay][iTrack];
        kfpart_decay.decayLengthErr   = decayLengthErr_KFDecayVertex[iDecay][iTrack];
        kfpart_decay.decayLengthXYErr = decayLengthXYErr_KFDecayVertex[iDecay][iTrack];
        kfpart_decay.lifeTimeErr      = lifeTimeErr_KFDecayVertex[iDecay][iTrack];
        kfpart_decay.rErr             = rErr_KFDecayVertex[iDecay][iTrack];

        kfpart_production.id   = iTrack;
        kfpart_production.idMC = mcpart_.trackID;
        kfpart_production.pdg  = pdg_KFProductionVertex[iDecay][iTrack];
        memcpy(kfpart_production.params, params_KFProductionVertex[iDecay], sizeof(kfpart_production.params));
        memcpy(kfpart_production.covMat, covMat_KFProductionVertex[iDecay], sizeof(kfpart_production.covMat));
        kfpart_production.isCovMatOK = isCovMatOK_KFProductionVertex[iDecay][iTrack];
        kfpart_production.x          = x_KFProductionVertex[iDecay][iTrack];
        kfpart_production.y          = y_KFProductionVertex[iDecay][iTrack];
        kfpart_production.z          = z_KFProductionVertex[iDecay][iTrack];
        kfpart_production.pX         = pX_KFProductionVertex[iDecay][iTrack];
        kfpart_production.pY         = pY_KFProductionVertex[iDecay][iTrack];
        kfpart_production.pZ         = pZ_KFProductionVertex[iDecay][iTrack];
        kfpart_production.e          = e_KFProductionVertex[iDecay][iTrack];
        kfpart_production.s          = s_KFProductionVertex[iDecay][iTrack];
        kfpart_production.charge     = charge_KFProductionVertex[iDecay][iTrack];
        kfpart_production.chi2       = chi2_KFProductionVertex[iDecay][iTrack];
        kfpart_production.ndf        = ndf_KFProductionVertex[iDecay][iTrack];
        kfpart_production.massHypo   = massHypo_KFProductionVertex[iDecay][iTrack];

        kfpart_production.isMother   = isMother_KFProductionVertex[iDecay][iTrack];
        kfpart_production.isDaughter = isDaughter_KFProductionVertex[iDecay][iTrack];

        kfpart_production.xErr          = xErr_KFProductionVertex[iDecay][iTrack];
        kfpart_production.yErr          = yErr_KFProductionVertex[iDecay][iTrack];
        kfpart_production.zErr          = zErr_KFProductionVertex[iDecay][iTrack];
        kfpart_production.pXErr         = pXErr_KFProductionVertex[iDecay][iTrack];
        kfpart_production.pYErr         = pYErr_KFProductionVertex[iDecay][iTrack];
        kfpart_production.pZErr         = pZErr_KFProductionVertex[iDecay][iTrack];
        kfpart_production.p             = pErr_KFProductionVertex[iDecay][iTrack];
        kfpart_production.pT            = pT_KFProductionVertex[iDecay][iTrack];
        kfpart_production.eta           = eta_KFProductionVertex[iDecay][iTrack];
        kfpart_production.phi           = phi_KFProductionVertex[iDecay][iTrack];
        kfpart_production.mass          = mass_KFProductionVertex[iDecay][iTrack];
        kfpart_production.decayLength   = decayLength_KFProductionVertex[iDecay][iTrack];
        kfpart_production.decayLengthXY = decayLengthXY_KFProductionVertex[iDecay][iTrack];
        kfpart_production.lifeTime      = lifeTime_KFProductionVertex[iDecay][iTrack];
        kfpart_production.r             = r_KFProductionVertex[iDecay][iTrack];
        kfpart_production.rapidity      = rapidity_KFProductionVertex[iDecay][iTrack];
        kfpart_production.theta         = theta_KFProductionVertex[iDecay][iTrack];

        kfpart_production.eErr             = eErr_KFProductionVertex[iDecay][iTrack];
        kfpart_production.sErr             = sErr_KFProductionVertex[iDecay][iTrack];
        kfpart_production.pErr             = pErr_KFProductionVertex[iDecay][iTrack];
        kfpart_production.pTErr            = pTErr_KFProductionVertex[iDecay][iTrack];
        kfpart_production.etaErr           = etaErr_KFProductionVertex[iDecay][iTrack];
        kfpart_production.phiErr           = phiErr_KFProductionVertex[iDecay][iTrack];
        kfpart_production.massErr          = massErr_KFProductionVertex[iDecay][iTrack];
        kfpart_production.decayLengthErr   = decayLengthErr_KFProductionVertex[iDecay][iTrack];
        kfpart_production.decayLengthXYErr = decayLengthXYErr_KFProductionVertex[iDecay][iTrack];
        kfpart_production.lifeTimeErr      = lifeTimeErr_KFProductionVertex[iDecay][iTrack];
        kfpart_production.rErr             = rErr_KFProductionVertex[iDecay][iTrack];

        for (auto& [name, obj] : output_map.map) {
          obj->Fill(iDecay, mcpart_, kfpart_decay, kfpart_production);
          obj->Fill(iDecay, kfpart_production);
          obj->Fill(iDecay, kfpart_decay);
          if (mcpart_.isMother) { obj->Fill(iDecay, mcpart_); }
          obj->Fill();
        }


      }  // end of track loop

    }  // end of KFDecayVertex event loop
  }  // end of decay loop
  ////////
  // end of event loop section
  ////

  output_map.PreWriteProcessAll();

  // Report sums of bin contents; for unweighted fills these are entry counts.
  const auto printMassRangeSummary = [](const TH1* hist) {
    const int nBins = hist->GetNbinsX();
    const double inside = hist->Integral(1, nBins);
    const double below = hist->GetBinContent(0);
    const double above = hist->GetBinContent(nBins + 1);
    const double outside = below + above;
    const double total = inside + outside;
    std::cout << "[Mass] " << hist->GetName()
              << " range=[" << hist->GetXaxis()->GetXmin()
              << ", " << hist->GetXaxis()->GetXmax() << ") GeV"
              << " inside=" << inside
              << " underflow=" << below
              << " overflow=" << above
              << " outside=" << outside
              << " (" << (total != 0. ? 100. * outside / total : 0.) << "%)"
              << " total=" << total << '\n';
  };






  /////////
  //  write all the stuff into output file
  ////
  cout << " Writing output file\n";

  for (int iDecay = 0; iDecay < NUM_OF_DECAYS; iDecay++) {
    MotherMCInitial_Dir[iDecay]->cd();
    fHistXMotherMCInitial[iDecay]->Write();
    fHistYMotherMCInitial[iDecay]->Write();
    fHistZMotherMCInitial[iDecay]->Write();
    fHistPxMotherMCInitial[iDecay]->Write();
    fHistPyMotherMCInitial[iDecay]->Write();
    fHistPzMotherMCInitial[iDecay]->Write();
    fHistChargeMotherMCInitial[iDecay]->Write();
    fHistPDGMotherMCInitial[iDecay]->Write();
    fHistPtMotherMCInitial[iDecay]->Write();
    fHistOneOverPtMotherMCInitial[iDecay]->Write();
    fHistPMotherMCInitial[iDecay]->Write();
    fHistEMotherMCInitial[iDecay]->Write();
    fHistMassMotherMCInitial[iDecay]->Write();
    fHistEtaMotherMCInitial[iDecay]->Write();
    fHistPhiMotherMCInitial[iDecay]->Write();

    for (int i = 0; i < NUM_OF_DAUGHTERS[iDecay]; i++) {
      DaughtersMCInitial_Dir[iDecay][i]->cd();
      fHistXDaughterMCInitial[iDecay][i]->Write();
      fHistYDaughterMCInitial[iDecay][i]->Write();
      fHistZDaughterMCInitial[iDecay][i]->Write();
      fHistPxDaughterMCInitial[iDecay][i]->Write();
      fHistPyDaughterMCInitial[iDecay][i]->Write();
      fHistPzDaughterMCInitial[iDecay][i]->Write();
      fHistChargeDaughterMCInitial[iDecay][i]->Write();
      fHistPDGDaughterMCInitial[iDecay][i]->Write();
      fHistPtDaughterMCInitial[iDecay][i]->Write();
      fHistOneOverPtDaughterMCInitial[iDecay][i]->Write();
      fHistPDaughterMCInitial[iDecay][i]->Write();
      fHistEDaughterMCInitial[iDecay][i]->Write();
      fHistMassDaughterMCInitial[iDecay][i]->Write();
      fHistEtaDaughterMCInitial[iDecay][i]->Write();
      fHistPhiDaughterMCInitial[iDecay][i]->Write();
    }

    MotherMCFinal_Dir[iDecay]->cd();
    fHistXMotherMCFinal[iDecay]->Write();
    fHistYMotherMCFinal[iDecay]->Write();
    fHistZMotherMCFinal[iDecay]->Write();
    fHistPxMotherMCFinal[iDecay]->Write();
    fHistPyMotherMCFinal[iDecay]->Write();
    fHistPzMotherMCFinal[iDecay]->Write();
    fHistChargeMotherMCFinal[iDecay]->Write();
    fHistPDGMotherMCFinal[iDecay]->Write();
    fHistPtMotherMCFinal[iDecay]->Write();
    fHistOneOverPtMotherMCFinal[iDecay]->Write();
    fHistPMotherMCFinal[iDecay]->Write();
    fHistEMotherMCFinal[iDecay]->Write();
    fHistMassMotherMCFinal[iDecay]->Write();
    fHistEtaMotherMCFinal[iDecay]->Write();
    fHistPhiMotherMCFinal[iDecay]->Write();

    for (int i = 0; i < NUM_OF_DAUGHTERS[iDecay]; i++) {
      DaughtersMCFinal_Dir[iDecay][i]->cd();
      fHistXDaughterMCFinal[iDecay][i]->Write();
      fHistYDaughterMCFinal[iDecay][i]->Write();
      fHistZDaughterMCFinal[iDecay][i]->Write();
      fHistPxDaughterMCFinal[iDecay][i]->Write();
      fHistPyDaughterMCFinal[iDecay][i]->Write();
      fHistPzDaughterMCFinal[iDecay][i]->Write();
      fHistChargeDaughterMCFinal[iDecay][i]->Write();
      fHistPDGDaughterMCFinal[iDecay][i]->Write();
      fHistPtDaughterMCFinal[iDecay][i]->Write();
      fHistOneOverPtDaughterMCFinal[iDecay][i]->Write();
      fHistPDaughterMCFinal[iDecay][i]->Write();
      fHistEDaughterMCFinal[iDecay][i]->Write();
      fHistMassDaughterMCFinal[iDecay][i]->Write();
      fHistEtaDaughterMCFinal[iDecay][i]->Write();
      fHistPhiDaughterMCFinal[iDecay][i]->Write();
    }

    MotherKFProductionVertex_Dir[iDecay]->cd();
    fHistXMotherKFProductionVertex[iDecay]->Write();
    fHistYMotherKFProductionVertex[iDecay]->Write();
    fHistZMotherKFProductionVertex[iDecay]->Write();
    fHistPxMotherKFProductionVertex[iDecay]->Write();
    fHistPyMotherKFProductionVertex[iDecay]->Write();
    fHistPzMotherKFProductionVertex[iDecay]->Write();
    fHistChargeMotherKFProductionVertex[iDecay]->Write();
    fHistPDGMotherKFProductionVertex[iDecay]->Write();
    fHistPtMotherKFProductionVertex[iDecay]->Write();
    fHistOneOverPtMotherKFProductionVertex[iDecay]->Write();
    fHistPMotherKFProductionVertex[iDecay]->Write();
    fHistEMotherKFProductionVertex[iDecay]->Write();
    fHistMassMotherKFProductionVertex[iDecay]->Write();
    printMassRangeSummary(fHistMassMotherKFProductionVertex[iDecay]);
    fHistEtaMotherKFProductionVertex[iDecay]->Write();
    fHistPhiMotherKFProductionVertex[iDecay]->Write();
    fHistChi2MotherKFProductionVertex[iDecay]->Write();
    fHistNDFMotherKFProductionVertex[iDecay]->Write();

    for (int i = 0; i < NUM_OF_DAUGHTERS[iDecay]; i++) {
      DaughtersKFProductionVertex_Dir[iDecay][i]->cd();
      fHistXDaughterKFProductionVertex[iDecay][i]->Write();
      fHistYDaughterKFProductionVertex[iDecay][i]->Write();
      fHistZDaughterKFProductionVertex[iDecay][i]->Write();
      fHistPxDaughterKFProductionVertex[iDecay][i]->Write();
      fHistPyDaughterKFProductionVertex[iDecay][i]->Write();
      fHistPzDaughterKFProductionVertex[iDecay][i]->Write();
      fHistChargeDaughterKFProductionVertex[iDecay][i]->Write();
      fHistPDGDaughterKFProductionVertex[iDecay][i]->Write();
      fHistPtDaughterKFProductionVertex[iDecay][i]->Write();
      fHistOneOverPtDaughterKFProductionVertex[iDecay][i]->Write();
      fHistPDaughterKFProductionVertex[iDecay][i]->Write();
      fHistEDaughterKFProductionVertex[iDecay][i]->Write();
      fHistMassDaughterKFProductionVertex[iDecay][i]->Write();
      printMassRangeSummary(fHistMassDaughterKFProductionVertex[iDecay][i]);
      fHistEtaDaughterKFProductionVertex[iDecay][i]->Write();
      fHistPhiDaughterKFProductionVertex[iDecay][i]->Write();
      fHistChi2DaughterKFProductionVertex[iDecay][i]->Write();
      fHistNDFDaughterKFProductionVertex[iDecay][i]->Write();
    }

    MotherKFDecayVertex_Dir[iDecay]->cd();
    fHistXMotherKFDecayVertex[iDecay]->Write();
    fHistYMotherKFDecayVertex[iDecay]->Write();
    fHistZMotherKFDecayVertex[iDecay]->Write();
    fHistPxMotherKFDecayVertex[iDecay]->Write();
    fHistPyMotherKFDecayVertex[iDecay]->Write();
    fHistPzMotherKFDecayVertex[iDecay]->Write();
    fHistChargeMotherKFDecayVertex[iDecay]->Write();
    fHistPDGMotherKFDecayVertex[iDecay]->Write();
    fHistPtMotherKFDecayVertex[iDecay]->Write();
    fHistOneOverPtMotherKFDecayVertex[iDecay]->Write();
    fHistPMotherKFDecayVertex[iDecay]->Write();
    fHistEMotherKFDecayVertex[iDecay]->Write();
    fHistMassMotherKFDecayVertex[iDecay]->Write();
    printMassRangeSummary(fHistMassMotherKFDecayVertex[iDecay]);
    fHistEtaMotherKFDecayVertex[iDecay]->Write();
    fHistPhiMotherKFDecayVertex[iDecay]->Write();
    fHistChi2MotherKFDecayVertex[iDecay]->Write();
    fHistNDFMotherKFDecayVertex[iDecay]->Write();

    for (int i = 0; i < NUM_OF_DAUGHTERS[iDecay]; i++) {
      DaughtersKFDecayVertex_Dir[iDecay][i]->cd();
      fHistXDaughterKFDecayVertex[iDecay][i]->Write();
      fHistYDaughterKFDecayVertex[iDecay][i]->Write();
      fHistZDaughterKFDecayVertex[iDecay][i]->Write();
      fHistPxDaughterKFDecayVertex[iDecay][i]->Write();
      fHistPyDaughterKFDecayVertex[iDecay][i]->Write();
      fHistPzDaughterKFDecayVertex[iDecay][i]->Write();
      fHistChargeDaughterKFDecayVertex[iDecay][i]->Write();
      fHistPDGDaughterKFDecayVertex[iDecay][i]->Write();
      fHistPtDaughterKFDecayVertex[iDecay][i]->Write();
      fHistOneOverPtDaughterKFDecayVertex[iDecay][i]->Write();
      fHistPDaughterKFDecayVertex[iDecay][i]->Write();
      fHistEDaughterKFDecayVertex[iDecay][i]->Write();
      fHistMassDaughterKFDecayVertex[iDecay][i]->Write();
      printMassRangeSummary(fHistMassDaughterKFDecayVertex[iDecay][i]);
      fHistEtaDaughterKFDecayVertex[iDecay][i]->Write();
      fHistPhiDaughterKFDecayVertex[iDecay][i]->Write();
      fHistChi2DaughterKFDecayVertex[iDecay][i]->Write();
      fHistNDFDaughterKFDecayVertex[iDecay][i]->Write();
    }
  }

  bool returnvalue = output_map.CheckAll();

  output_map.WriteAll();

  OtherHistos->cd();
  fHistPtPtErrMother->Write();

  outputFile.Close();

  cout << endl;
  cout << "The macro is done. Results are saved in " << KFHistoOutputFile << endl;

  if (returnvalue) { cout << "Macro finished successfully" << endl; }
  if (!returnvalue) { cout << "Macro finished with errors" << endl; }

  for (int i = 0; i < NUM_OF_DECAYS; i++) {
    std::cout << "Decay: " << MOTHER_PDG_IDS[i] << " has " << event_counts[i] << " events" << std::endl;
  }
}
