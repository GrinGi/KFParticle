/*
OutputObjInterface-inherited class
to create, fill and write QA histograms of
the Transport(...) method check
*/

#pragma once

#include "OutputObjInterface.h"

class TransportMethodCheck : public OutputObjInterface {
 public:
  TransportMethodCheck(TFile& outputFile) : OutputObjInterface(outputFile) {};
  ~TransportMethodCheck() {};

  // methods that must be realized by user
  void Create() override;
  void Fill() override {};
  void Fill(int iDecay, const MCParticleStruct& mcPart) override;
  void Fill(int iDecay, const KFParticleStruct& kfPart) override;
  void Fill(int iDecay, const MCParticleStruct& mcPart, const KFParticleStruct& kfPart) override;
  void Fill(int iDecay, const MCParticleStruct& mcPart, const KFParticleStruct& kfPartDecay,
            const KFParticleStruct& kfPartProduction) override;
  void PreWriteProcess() override;
  void Write() override;
  bool Check() override;

 private:
  // user-defined histograms
  std::array<TH1F*, NUM_OF_DECAYS> fHistTrackLength;

  std::array<TH1F*, NUM_OF_DECAYS> fHistDeltaXAbs;
  std::array<TH1F*, NUM_OF_DECAYS> fHistDeltaYAbs;
  std::array<TH1F*, NUM_OF_DECAYS> fHistDeltaZAbs;
  std::array<TH1F*, NUM_OF_DECAYS> fHistDeltaRAbs;
  std::array<TH1F*, NUM_OF_DECAYS> fHistDeltaPXAbs;
  std::array<TH1F*, NUM_OF_DECAYS> fHistDeltaPYAbs;
  std::array<TH1F*, NUM_OF_DECAYS> fHistDeltaPZAbs;
  std::array<TH1F*, NUM_OF_DECAYS> fHistDeltaPTAbs;
  std::array<TH1F*, NUM_OF_DECAYS> fHistDeltaPAbs;
  std::array<TH1F*, NUM_OF_DECAYS> fHistDeltaEAbs;
  std::array<TH1F*, NUM_OF_DECAYS> fHistDeltaMassAbs;

  std::array<TH1F*, NUM_OF_DECAYS> fHistDeltaXOverTrackLength;
  std::array<TH1F*, NUM_OF_DECAYS> fHistDeltaYOverTrackLength;
  std::array<TH1F*, NUM_OF_DECAYS> fHistDeltaZOverTrackLength;
  std::array<TH1F*, NUM_OF_DECAYS> fHistDeltaROverTrackLength;
  std::array<TH1F*, NUM_OF_DECAYS> fHistDeltaPXOverTrackLength;
  std::array<TH1F*, NUM_OF_DECAYS> fHistDeltaPYOverTrackLength;
  std::array<TH1F*, NUM_OF_DECAYS> fHistDeltaPZOverTrackLength;
  std::array<TH1F*, NUM_OF_DECAYS> fHistDeltaPTOverTrackLength;
  std::array<TH1F*, NUM_OF_DECAYS> fHistDeltaPOverTrackLength;
  std::array<TH1F*, NUM_OF_DECAYS> fHistDeltaEOverTrackLength;
  std::array<TH1F*, NUM_OF_DECAYS> fHistDeltaMassOverTrackLength;

  std::array<TH2F*, NUM_OF_DECAYS> fHistDeltaXvsTrackLength;
  std::array<TH2F*, NUM_OF_DECAYS> fHistDeltaYvsTrackLength;
  std::array<TH2F*, NUM_OF_DECAYS> fHistDeltaZvsTrackLength;
  std::array<TH2F*, NUM_OF_DECAYS> fHistDeltaRvsTrackLength;
  std::array<TH2F*, NUM_OF_DECAYS> fHistDeltaPXvsTrackLength;
  std::array<TH2F*, NUM_OF_DECAYS> fHistDeltaPYvsTrackLength;
  std::array<TH2F*, NUM_OF_DECAYS> fHistDeltaPZvsTrackLength;
  std::array<TH2F*, NUM_OF_DECAYS> fHistDeltaPTvsTrackLength;
  std::array<TH2F*, NUM_OF_DECAYS> fHistDeltaPvsTrackLength;
  std::array<TH2F*, NUM_OF_DECAYS> fHistDeltaEvsTrackLength;
  std::array<TH2F*, NUM_OF_DECAYS> fHistDeltaMassvsTrackLength;
};

////---------- method implementations ----------

void TransportMethodCheck::Create()
{
  for (int iDecay = 0; iDecay < NUM_OF_DECAYS; iDecay++) {
    fHistTrackLength[iDecay] =
      new TH1F(Form("fHistTrackLength_%i", MOTHER_PDG_IDS[iDecay]), "Track Length;Lmc, cm; counts", 50, 0, 0.0000005);

    fHistDeltaXAbs[iDecay] =
      new TH1F(Form("fHistDeltaXAbs_%i", MOTHER_PDG_IDS[iDecay]), "#Delta X;#Delta X, cm; counts", 50, -0.0001, 0.0001);
    fHistDeltaYAbs[iDecay] =
      new TH1F(Form("fHistDeltaYAbs_%i", MOTHER_PDG_IDS[iDecay]), "#Delta Y;#Delta Y, cm; counts", 50, -0.0001, 0.0001);
    fHistDeltaZAbs[iDecay] =
      new TH1F(Form("fHistDeltaZAbs_%i", MOTHER_PDG_IDS[iDecay]), "#Delta Z;#Delta Z, cm; counts", 50, -0.0001, 0.0001);
    fHistDeltaRAbs[iDecay] =
      new TH1F(Form("fHistDeltaRAbs_%i", MOTHER_PDG_IDS[iDecay]), "#Delta R;#Delta R, cm; counts", 50, -5, 5);
    fHistDeltaPXAbs[iDecay] =
      new TH1F(Form("fHistDeltaPXAbs_%i", MOTHER_PDG_IDS[iDecay]), "#Delta Px;#Delta Px, GeV; counts", 50, -1, 1);
    fHistDeltaPYAbs[iDecay] =
      new TH1F(Form("fHistDeltaPYAbs_%i", MOTHER_PDG_IDS[iDecay]), "#Delta Py;#Delta Py, GeV; counts", 50, -1, 1);
    fHistDeltaPZAbs[iDecay] =
      new TH1F(Form("fHistDeltaPZAbs_%i", MOTHER_PDG_IDS[iDecay]), "#Delta Pz;#Delta Pz, GeV; counts", 50, -0.1, 0.1);
    fHistDeltaPTAbs[iDecay] =
      new TH1F(Form("fHistDeltaPTAbs_%i", MOTHER_PDG_IDS[iDecay]), "#Delta Pt;#Delta Pt, GeV; counts", 50, -0.1, 0.1);
    fHistDeltaPAbs[iDecay] =
      new TH1F(Form("fHistDeltaPAbs_%i", MOTHER_PDG_IDS[iDecay]), "#Delta P;#Delta P, GeV; counts", 50, -0.1, 0.1);
    fHistDeltaEAbs[iDecay] =
      new TH1F(Form("fHistDeltaEAbs_%i", MOTHER_PDG_IDS[iDecay]), "#Delta E;#Delta E, GeV; counts", 50, -0.1, 0.1);
    fHistDeltaMassAbs[iDecay] = new TH1F(Form("fHistDeltaMassAbs_%i", MOTHER_PDG_IDS[iDecay]),
                                         "#Delta Mass;#Delta Mass, GeV; counts", 50, -0.1, 0.1);

    fHistDeltaXOverTrackLength[iDecay]  = new TH1F(Form("fHistDeltaXOverTrackLength_%i", MOTHER_PDG_IDS[iDecay]),
                                                   "#Delta X/Lmc; #Delta X/Lmc; counts", 50, -0.1, 0.1);
    fHistDeltaYOverTrackLength[iDecay]  = new TH1F(Form("fHistDeltaYOverTrackLength_%i", MOTHER_PDG_IDS[iDecay]),
                                                   "#Delta Y/Lmc; #Delta Y/Lmc; counts", 50, -0.1, 0.1);
    fHistDeltaZOverTrackLength[iDecay]  = new TH1F(Form("fHistDeltaZOverTrackLength_%i", MOTHER_PDG_IDS[iDecay]),
                                                   "#Delta Z/Lmc; #Delta Z/Lmc; counts", 50, -1, 1);
    fHistDeltaROverTrackLength[iDecay]  = new TH1F(Form("fHistDeltaROverTrackLength_%i", MOTHER_PDG_IDS[iDecay]),
                                                   "#Delta R/Lmc; #Delta R/Lmc; counts", 50, -1, 1);
    fHistDeltaPXOverTrackLength[iDecay] = new TH1F(Form("fHistDeltaPXOverTrackLength_%i", MOTHER_PDG_IDS[iDecay]),
                                                   "#Delta Px/Lmc; #Delta Px/Lmc, GeV/cm; counts", 50, -1e-3, 1e-3);
    fHistDeltaPYOverTrackLength[iDecay] = new TH1F(Form("fHistDeltaPYOverTrackLength_%i", MOTHER_PDG_IDS[iDecay]),
                                                   "#Delta Py/Lmc; #Delta Py/Lmc, GeV/cm; counts", 50, -1e-3, 1e-3);
    fHistDeltaPZOverTrackLength[iDecay] = new TH1F(Form("fHistDeltaPZOverTrackLength_%i", MOTHER_PDG_IDS[iDecay]),
                                                   "#Delta Pz/Lmc; #Delta Pz/Lmc, GeV/cm; counts", 50, -1e-3, 1e-3);
    fHistDeltaPTOverTrackLength[iDecay] = new TH1F(Form("fHistDeltaPTOverTrackLength_%i", MOTHER_PDG_IDS[iDecay]),
                                                   "#Delta Pt/Lmc; #Delta Pt/Lmc, GeV/cm; counts", 50, -1e-3, 1e-3);
    fHistDeltaPOverTrackLength[iDecay]  = new TH1F(Form("fHistDeltaPOverTrackLength_%i", MOTHER_PDG_IDS[iDecay]),
                                                   "#Delta P/Lmc; #Delta P/Lmc, GeV/cm; counts", 50, -1e-3, 1e-3);
    fHistDeltaEOverTrackLength[iDecay]  = new TH1F(Form("fHistDeltaEOverTrackLength_%i", MOTHER_PDG_IDS[iDecay]),
                                                   "#Delta E/Lmc; #Delta E/Lmc, GeV/cm; counts", 50, -1e-2, 1e-2);
    fHistDeltaMassOverTrackLength[iDecay] =
      new TH1F(Form("fHistDeltaMassOverTrackLength_%i", MOTHER_PDG_IDS[iDecay]),
               "#Delta Mass/Lmc, GeV/cm; #Delta Mass/Lmc; counts", 50, -1e-2, 1e-2);

    fHistDeltaXvsTrackLength[iDecay] =
      new TH2F(Form("fHistDeltaXvsTrackLength_%i", MOTHER_PDG_IDS[iDecay]), "#Delta X vs Lmc; #Delta X, cm; Lmc, cm",
               50, -0.00001, 0.00001, 50, 0, 100);
    fHistDeltaYvsTrackLength[iDecay] =
      new TH2F(Form("fHistDeltaYvsTrackLength_%i", MOTHER_PDG_IDS[iDecay]), "#Delta Y vs Lmc; #Delta Y, cm; Lmc, cm",
               50, -0.0001, 0.0001, 50, 0, 100);
    fHistDeltaZvsTrackLength[iDecay] =
      new TH2F(Form("fHistDeltaZvsTrackLength_%i", MOTHER_PDG_IDS[iDecay]), "#Delta Z vs Lmc; #Delta Z, cm; Lmc, cm",
               50, -0.0001, 0.0001, 50, 0, 100);
    fHistDeltaRvsTrackLength[iDecay]  = new TH2F(Form("fHistDeltaRvsTrackLength_%i", MOTHER_PDG_IDS[iDecay]),
                                                 "#Delta R vs Lmc; #Delta R, cm; Lmc, cm", 50, -5, 5, 50, 0, 100);
    fHistDeltaPXvsTrackLength[iDecay] = new TH2F(Form("fHistDeltaPXvsTrackLength_%i", MOTHER_PDG_IDS[iDecay]),
                                                 "#Delta Px vs Lmc; #Delta Px, GeV; Lmc, cm", 50, -1, 1, 50, 0, 100);
    fHistDeltaPYvsTrackLength[iDecay] = new TH2F(Form("fHistDeltaPYvsTrackLength_%i", MOTHER_PDG_IDS[iDecay]),
                                                 "#Delta Py vs Lmc; #Delta Py, GeV; Lmc, cm", 50, -1, 1, 50, 0, 100);
    fHistDeltaPZvsTrackLength[iDecay] =
      new TH2F(Form("fHistDeltaPZvsTrackLength_%i", MOTHER_PDG_IDS[iDecay]),
               "#Delta Pz vs Lmc; #Delta Pz, GeV; Lmc, cm", 50, -0.1, 0.1, 50, 0, 100);
    fHistDeltaPTvsTrackLength[iDecay] =
      new TH2F(Form("fHistDeltaPTvsTrackLength_%i", MOTHER_PDG_IDS[iDecay]),
               "#Delta Pt vs Lmc; #Delta Pt, GeV; Lmc, cm", 50, -0.1, 0.1, 50, 0, 100);
    fHistDeltaPvsTrackLength[iDecay] = new TH2F(Form("fHistDeltaPvsTrackLength_%i", MOTHER_PDG_IDS[iDecay]),
                                                "#Delta P vs Lmc; #Delta P, GeV; Lmc, cm", 50, -0.1, 0.1, 50, 0, 100);
    fHistDeltaEvsTrackLength[iDecay] = new TH2F(Form("fHistDeltaEvsTrackLength_%i", MOTHER_PDG_IDS[iDecay]),
                                                "#Delta E vs Lmc; #Delta E, GeV; Lmc, cm", 50, -0.1, 0.1, 50, 0, 100);
    fHistDeltaMassvsTrackLength[iDecay] =
      new TH2F(Form("fHistDeltaMassvsTrackLength_%i", MOTHER_PDG_IDS[iDecay]),
               "#Delta Mass vs Lmc; #Delta Mass, GeV; Lmc, cm", 50, -0.1, 0.1, 50, 0, 100);
  }
}

void TransportMethodCheck::Fill(int iDecay, const MCParticleStruct& mcPart, const KFParticleStruct& kfPartDecay,
                                const KFParticleStruct& kfPartProduction) {};

void TransportMethodCheck::Fill(int iDecay, const MCParticleStruct& mcPart)
{
  // MCParticleStruct part(mcPart);
  KFParticle kfPart;
  // kfPart.SetConstructMethod(2);
  float params[6];
  params[0] = mcPart.initialX;
  params[1] = mcPart.initialY;
  params[2] = mcPart.initialZ;
  params[3] = mcPart.initialPX;
  params[4] = mcPart.initialPY;
  params[5] = mcPart.initialPZ;
  // auto covmat_vec_ = MakeCovMatALICE(mcPart);
  float covmat[21];
  // std::copy(covmat_vec_.begin(), covmat_vec_.end(), covmat);
  auto charge_ = mcPart.charge;
  auto chi2_   = 0;  // mcPart.chi2;
  auto ndf_    = 0;  // mcPart.ndf;
  auto mass_   = mcPart.mass;
#ifdef TESTSUITE
  kfPart.Initialize(params, covmat, charge_, chi2_, ndf_, mass_);
#else
  kfPart.Create(params, covmat, charge_, mass_);
#endif
  kfPart.SetId(mcPart.trackID);
  kfPart.SetPDG(mcPart.pdg);
  // kfPart.SetAtProductionVertex(true);

  if (mcPart.trackLength < 0.00000005) { return; }

  float xyz[3] = {mcPart.finalX, mcPart.finalY, mcPart.finalZ};
  kfPart.TransportToPoint(xyz); /**/

  float _mass_, _mass_err_;
  float _pt_, _pt_err_;
  float _p_, _p_err_;
  kfPart.GetMass(_mass_, _mass_err_);
  kfPart.GetPt(_pt_, _pt_err_);
  kfPart.GetMomentum(_p_, _p_err_);

  float deltaR = sqrt(pow(mcPart.finalX - kfPart.GetX(), 2) + pow(mcPart.finalY - kfPart.GetY(), 2)
                      + pow(mcPart.finalZ - kfPart.GetZ(), 2));

  fHistTrackLength[iDecay]->Fill(mcPart.trackLength);

  fHistDeltaXAbs[iDecay]->Fill(mcPart.finalX - kfPart.GetX());
  fHistDeltaYAbs[iDecay]->Fill(mcPart.finalY - kfPart.GetY());
  fHistDeltaZAbs[iDecay]->Fill(mcPart.finalZ - kfPart.GetZ());
  fHistDeltaRAbs[iDecay]->Fill(deltaR);
  fHistDeltaPXAbs[iDecay]->Fill(mcPart.finalPX - kfPart.GetPx());
  fHistDeltaPYAbs[iDecay]->Fill(mcPart.finalPY - kfPart.GetPy());
  fHistDeltaPZAbs[iDecay]->Fill(mcPart.finalPZ - kfPart.GetPz());
  fHistDeltaPTAbs[iDecay]->Fill(mcPart.finalPT() - _pt_);
  fHistDeltaPAbs[iDecay]->Fill(mcPart.finalP() - _p_);
  fHistDeltaEAbs[iDecay]->Fill(mcPart.finalE() - kfPart.GetE());
  fHistDeltaMassAbs[iDecay]->Fill(mcPart.mass - _mass_);

  fHistDeltaXOverTrackLength[iDecay]->Fill((mcPart.finalX - kfPart.GetX()) / mcPart.trackLength);
  fHistDeltaYOverTrackLength[iDecay]->Fill((mcPart.finalY - kfPart.GetY()) / mcPart.trackLength);
  fHistDeltaZOverTrackLength[iDecay]->Fill((mcPart.finalZ - kfPart.GetZ()) / mcPart.trackLength);
  fHistDeltaROverTrackLength[iDecay]->Fill(deltaR / mcPart.trackLength);
  fHistDeltaPXOverTrackLength[iDecay]->Fill((mcPart.finalPX - kfPart.GetPx()) / mcPart.trackLength);
  fHistDeltaPYOverTrackLength[iDecay]->Fill((mcPart.finalPY - kfPart.GetPy()) / mcPart.trackLength);
  fHistDeltaPZOverTrackLength[iDecay]->Fill((mcPart.finalPZ - kfPart.GetPz()) / mcPart.trackLength);
  fHistDeltaPTOverTrackLength[iDecay]->Fill((mcPart.finalPT() - _pt_) / mcPart.trackLength);
  fHistDeltaPOverTrackLength[iDecay]->Fill((mcPart.finalP() - _p_) / mcPart.trackLength);
  fHistDeltaEOverTrackLength[iDecay]->Fill((mcPart.finalE() - kfPart.GetE()) / mcPart.trackLength);
  fHistDeltaMassOverTrackLength[iDecay]->Fill((mcPart.mass - _mass_) / mcPart.trackLength);

  fHistDeltaXvsTrackLength[iDecay]->Fill(mcPart.finalX - kfPart.GetX(), mcPart.trackLength);
  fHistDeltaYvsTrackLength[iDecay]->Fill(mcPart.finalY - kfPart.GetY(), mcPart.trackLength);
  fHistDeltaZvsTrackLength[iDecay]->Fill(mcPart.finalZ - kfPart.GetZ(), mcPart.trackLength);
  fHistDeltaRvsTrackLength[iDecay]->Fill(deltaR, mcPart.trackLength);
  fHistDeltaPXvsTrackLength[iDecay]->Fill(mcPart.finalPX - kfPart.GetPx(), mcPart.trackLength);
  fHistDeltaPYvsTrackLength[iDecay]->Fill(mcPart.finalPY - kfPart.GetPy(), mcPart.trackLength);
  fHistDeltaPZvsTrackLength[iDecay]->Fill(mcPart.finalPZ - kfPart.GetPz(), mcPart.trackLength);
  fHistDeltaPTvsTrackLength[iDecay]->Fill(mcPart.finalPT() - _pt_, mcPart.trackLength);
  fHistDeltaPvsTrackLength[iDecay]->Fill(mcPart.finalP() - _p_, mcPart.trackLength);
  fHistDeltaEvsTrackLength[iDecay]->Fill(mcPart.finalE() - kfPart.GetE(), mcPart.trackLength);
  fHistDeltaMassvsTrackLength[iDecay]->Fill(mcPart.mass - _mass_, mcPart.trackLength);

  // cout << mcPart.trackID << " " << kfPart.GetMass() << "\n";
}

void TransportMethodCheck::Fill(int iDecay, const KFParticleStruct& kfPart)
{
  //....
}

void TransportMethodCheck::Fill(int iDecay, const MCParticleStruct& mcPart, const KFParticleStruct& kfPart)
{
  //....
}

void TransportMethodCheck::PreWriteProcess() {}

bool TransportMethodCheck::Check() { return true; }

void TransportMethodCheck::Write()
{
  TDirectory* TransportMethodCheck_Dir = outputFile->mkdir("TransportMethodCheck");
  std::array<TDirectory*, NUM_OF_DECAYS> TransportDeltasAbs_Dir;
  std::array<TDirectory*, NUM_OF_DECAYS> TransportDeltasOverTrackLength_Dir;
  std::array<TDirectory*, NUM_OF_DECAYS> TransportDeltasVsTrackLength_Dir;

  for (int iDecay = 0; iDecay < NUM_OF_DECAYS; iDecay++) {

    TransportMethodCheck_Dir->cd();
    fHistTrackLength[iDecay]->Write();

    TransportDeltasAbs_Dir[iDecay] = TransportMethodCheck_Dir->mkdir(Form("Deltas_%i", MOTHER_PDG_IDS[iDecay]));

    TransportDeltasAbs_Dir[iDecay]->cd();
    fHistDeltaXAbs[iDecay]->Write();
    fHistDeltaYAbs[iDecay]->Write();
    fHistDeltaZAbs[iDecay]->Write();
    fHistDeltaRAbs[iDecay]->Write();
    fHistDeltaPXAbs[iDecay]->Write();
    fHistDeltaPYAbs[iDecay]->Write();
    fHistDeltaPZAbs[iDecay]->Write();
    fHistDeltaPTAbs[iDecay]->Write();
    fHistDeltaPAbs[iDecay]->Write();
    fHistDeltaEAbs[iDecay]->Write();
    fHistDeltaMassAbs[iDecay]->Write();

    TransportDeltasOverTrackLength_Dir[iDecay] =
      TransportMethodCheck_Dir->mkdir(Form("DeltasOverTrackLength_%i", MOTHER_PDG_IDS[iDecay]));

    TransportDeltasOverTrackLength_Dir[iDecay]->cd();
    fHistDeltaXOverTrackLength[iDecay]->Write();
    fHistDeltaYOverTrackLength[iDecay]->Write();
    fHistDeltaZOverTrackLength[iDecay]->Write();
    fHistDeltaROverTrackLength[iDecay]->Write();
    fHistDeltaPXOverTrackLength[iDecay]->Write();
    fHistDeltaPYOverTrackLength[iDecay]->Write();
    fHistDeltaPZOverTrackLength[iDecay]->Write();
    fHistDeltaPTOverTrackLength[iDecay]->Write();
    fHistDeltaPOverTrackLength[iDecay]->Write();
    fHistDeltaEOverTrackLength[iDecay]->Write();
    fHistDeltaMassOverTrackLength[iDecay]->Write();

    TransportDeltasVsTrackLength_Dir[iDecay] =
      TransportMethodCheck_Dir->mkdir(Form("DeltasVsTrackLength_%i", MOTHER_PDG_IDS[iDecay]));
    TransportDeltasVsTrackLength_Dir[iDecay]->cd();
    fHistDeltaXvsTrackLength[iDecay]->Write();
    fHistDeltaYvsTrackLength[iDecay]->Write();
    fHistDeltaZvsTrackLength[iDecay]->Write();
    fHistDeltaRvsTrackLength[iDecay]->Write();
    fHistDeltaPXvsTrackLength[iDecay]->Write();
    fHistDeltaPYvsTrackLength[iDecay]->Write();
    fHistDeltaPZvsTrackLength[iDecay]->Write();
    fHistDeltaPTvsTrackLength[iDecay]->Write();
    fHistDeltaPvsTrackLength[iDecay]->Write();
    fHistDeltaEvsTrackLength[iDecay]->Write();
    fHistDeltaMassvsTrackLength[iDecay]->Write();
  }
}