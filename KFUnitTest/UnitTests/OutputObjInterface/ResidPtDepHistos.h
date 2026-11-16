#pragma once

#include "../ConfigConstants.h"
#include "MakeGraphsFromTH2.h"
#include "OutputObjInterface.h"
#include "TCanvas.h"
#include "TH2.h"
#include <array>

class ResidPtDepHistos : public OutputObjInterface {
 public:
  ResidPtDepHistos(TFile& outputFile) : OutputObjInterface(outputFile) {};
  ~ResidPtDepHistos() {};

  void Create() override;
  void Fill() override {};
  void Fill(int iDecay, const MCParticleStruct& mcPart) override;
  void Fill(int iDecay, const KFParticleStruct& kfPart) override;
  void Fill(int iDecay, const MCParticleStruct& mcPart, const KFParticleStruct& kfPartDecay,
            const KFParticleStruct& kfPartProduction) override;
  void Fill(int iDecay, const MCParticleStruct& mcPart, const KFParticleStruct& kfPart) override;
  void PreWriteProcess() override;
  void Write() override;
  bool Check() override;

 private:
  TH2F* fHistPtXResidMotherFst[NUM_OF_DECAYS];
  TH2F* fHistPtYResidMotherFst[NUM_OF_DECAYS];
  TH2F* fHistPtZResidMotherFst[NUM_OF_DECAYS];
  TH2F* fHistPtPxResidMotherFst[NUM_OF_DECAYS];
  TH2F* fHistPtPyResidMotherFst[NUM_OF_DECAYS];
  TH2F* fHistPtPzResidMotherFst[NUM_OF_DECAYS];
  TH2F* fHistPtPtResidMotherFst[NUM_OF_DECAYS];
  TH2F* fHistPtMassResidMotherFst[NUM_OF_DECAYS];
  TH2F* fHistPtEResidMotherFst[NUM_OF_DECAYS];

  TH2F* fHistPtXResidMotherLst[NUM_OF_DECAYS];
  TH2F* fHistPtYResidMotherLst[NUM_OF_DECAYS];
  TH2F* fHistPtZResidMotherLst[NUM_OF_DECAYS];
  TH2F* fHistPtPxResidMotherLst[NUM_OF_DECAYS];
  TH2F* fHistPtPyResidMotherLst[NUM_OF_DECAYS];
  TH2F* fHistPtPzResidMotherLst[NUM_OF_DECAYS];
  TH2F* fHistPtPtResidMotherLst[NUM_OF_DECAYS];
  TH2F* fHistPtMassResidMotherLst[NUM_OF_DECAYS];
  TH2F* fHistPtEResidMotherLst[NUM_OF_DECAYS];

  std::array<std::array<TH2F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistPtXResidDaughter;
  std::array<std::array<TH2F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistPtYResidDaughter;
  std::array<std::array<TH2F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistPtZResidDaughter;
  std::array<std::array<TH2F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistPtPxResidDaughter;
  std::array<std::array<TH2F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistPtPyResidDaughter;
  std::array<std::array<TH2F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistPtPzResidDaughter;
  std::array<std::array<TH2F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistPtPtResidDaughter;
  std::array<std::array<TH2F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistPtMassResidDaughter;
  std::array<std::array<TH2F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistPtEResidDaughter;

  //
  TH1D* fHistXResidMotherFst[NUM_OF_DECAYS];
  TH1D* fHistYResidMotherFst[NUM_OF_DECAYS];
  TH1D* fHistZResidMotherFst[NUM_OF_DECAYS];
  TH1D* fHistPxResidMotherFst[NUM_OF_DECAYS];
  TH1D* fHistPyResidMotherFst[NUM_OF_DECAYS];
  TH1D* fHistPzResidMotherFst[NUM_OF_DECAYS];
  TH1D* fHistPtResidMotherFst[NUM_OF_DECAYS];
  TH1D* fHistMassResidMotherFst[NUM_OF_DECAYS];
  TH1D* fHistEResidMotherFst[NUM_OF_DECAYS];

  TH1D* fHistXResidMotherLst[NUM_OF_DECAYS];
  TH1D* fHistYResidMotherLst[NUM_OF_DECAYS];
  TH1D* fHistZResidMotherLst[NUM_OF_DECAYS];
  TH1D* fHistPxResidMotherLst[NUM_OF_DECAYS];
  TH1D* fHistPyResidMotherLst[NUM_OF_DECAYS];
  TH1D* fHistPzResidMotherLst[NUM_OF_DECAYS];
  TH1D* fHistPtResidMotherLst[NUM_OF_DECAYS];
  TH1D* fHistMassResidMotherLst[NUM_OF_DECAYS];
  TH1D* fHistEResidMotherLst[NUM_OF_DECAYS];

  std::array<std::array<TH1D*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistXResidDaughter;
  std::array<std::array<TH1D*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistYResidDaughter;
  std::array<std::array<TH1D*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistZResidDaughter;
  std::array<std::array<TH1D*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistPxResidDaughter;
  std::array<std::array<TH1D*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistPyResidDaughter;
  std::array<std::array<TH1D*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistPzResidDaughter;
  std::array<std::array<TH1D*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistPtResidDaughter;
  std::array<std::array<TH1D*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistMassResidDaughter;
  std::array<std::array<TH1D*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistEResidDaughter;

  //
  TGraphErrors* fGraphPtXMeanResidMother[NUM_OF_DECAYS];
  TGraphErrors* fGraphPtYMeanResidMother[NUM_OF_DECAYS];
  TGraphErrors* fGraphPtZMeanResidMother[NUM_OF_DECAYS];
  TGraphErrors* fGraphPtPxMeanResidMother[NUM_OF_DECAYS];
  TGraphErrors* fGraphPtPyMeanResidMother[NUM_OF_DECAYS];
  TGraphErrors* fGraphPtPzMeanResidMother[NUM_OF_DECAYS];
  TGraphErrors* fGraphPtPtMeanResidMother[NUM_OF_DECAYS];
  TGraphErrors* fGraphPtMassMeanResidMother[NUM_OF_DECAYS];
  TGraphErrors* fGraphPtEMeanResidMother[NUM_OF_DECAYS];

  std::array<std::array<TGraphErrors*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fGraphPtXMeanResidDaughter;
  std::array<std::array<TGraphErrors*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fGraphPtYMeanResidDaughter;
  std::array<std::array<TGraphErrors*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fGraphPtZMeanResidDaughter;
  std::array<std::array<TGraphErrors*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fGraphPtPxMeanResidDaughter;
  std::array<std::array<TGraphErrors*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fGraphPtPyMeanResidDaughter;
  std::array<std::array<TGraphErrors*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fGraphPtPzMeanResidDaughter;
  std::array<std::array<TGraphErrors*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fGraphPtPtMeanResidDaughter;
  std::array<std::array<TGraphErrors*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fGraphPtMassMeanResidDaughter;
  std::array<std::array<TGraphErrors*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fGraphPtEMeanResidDaughter;

  //
  TGraphErrors* fGraphPtXResolMother[NUM_OF_DECAYS];
  TGraphErrors* fGraphPtYResolMother[NUM_OF_DECAYS];
  TGraphErrors* fGraphPtZResolMother[NUM_OF_DECAYS];
  TGraphErrors* fGraphPtPxResolMother[NUM_OF_DECAYS];
  TGraphErrors* fGraphPtPyResolMother[NUM_OF_DECAYS];
  TGraphErrors* fGraphPtPzResolMother[NUM_OF_DECAYS];
  TGraphErrors* fGraphPtPtResolMother[NUM_OF_DECAYS];
  TGraphErrors* fGraphPtMassResolMother[NUM_OF_DECAYS];
  TGraphErrors* fGraphPtEResolMother[NUM_OF_DECAYS];

  std::array<std::array<TGraphErrors*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fGraphPtXResolDaughter;
  std::array<std::array<TGraphErrors*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fGraphPtYResolDaughter;
  std::array<std::array<TGraphErrors*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fGraphPtZResolDaughter;
  std::array<std::array<TGraphErrors*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fGraphPtPxResolDaughter;
  std::array<std::array<TGraphErrors*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fGraphPtPyResolDaughter;
  std::array<std::array<TGraphErrors*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fGraphPtPzResolDaughter;
  std::array<std::array<TGraphErrors*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fGraphPtPtResolDaughter;
  std::array<std::array<TGraphErrors*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fGraphPtMassResolDaughter;
  std::array<std::array<TGraphErrors*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fGraphPtEResolDaughter;
};

////////////////////////////////////////////////
////---------- method implementations ----------
////////////////////////////////////////////////
void ResidPtDepHistos::Create()
{
  int LogBinsX       = 30;
  double StartValueX = 0.05;
  double StopValueX  = 12;
  double LogWidthX[LogBinsX + 1];
  for (int i = 0; i <= LogBinsX; i++) {
    LogWidthX[i] =
      pow(10, log10(StartValueX) + (log10(StopValueX) - log10(StartValueX)) / double(LogBinsX) * double(i));
  }

  for (int iDecay = 0; iDecay < NUM_OF_DECAYS; iDecay++) {
    fHistPtXResidMotherFst[iDecay] =
      new TH2F(Form("fHistPtXResidMotherFst_%i", MOTHER_PDG_IDS[iDecay]),
               "Pt-X Resid histo; p_{T}, GeV; X residual, cm", LogBinsX, LogWidthX, 100, -1e-1, 1e-1);
    fHistPtYResidMotherFst[iDecay] =
      new TH2F(Form("fHistPtYResidMotherFst_%i", MOTHER_PDG_IDS[iDecay]),
               "Pt-Y Resid histo; p_{T}, GeV; Y residual, cm", LogBinsX, LogWidthX, 100, -1e-1, 1e-1);
    fHistPtZResidMotherFst[iDecay] =
      new TH2F(Form("fHistPtZResidMotherFst_%i", MOTHER_PDG_IDS[iDecay]),
               "Pt-Z Resid histo; p_{T}, GeV; Z residual, cm", LogBinsX, LogWidthX, 100, -1e-1, 1e-1);
    fHistPtPxResidMotherFst[iDecay] =
      new TH2F(Form("fHistPtPxResidMotherFst_%i", MOTHER_PDG_IDS[iDecay]),
               "Pt-Px Resid histo; p_{T}, GeV; Px residual, GeV", LogBinsX, LogWidthX, 50, -0.15, 0.15);
    fHistPtPyResidMotherFst[iDecay] =
      new TH2F(Form("fHistPtPyResidMotherFst_%i", MOTHER_PDG_IDS[iDecay]),
               "Pt-Py Resid histo; p_{T}, GeV; Py residual, GeV", LogBinsX, LogWidthX, 50, -0.15, 0.15);
    fHistPtPzResidMotherFst[iDecay] =
      new TH2F(Form("fHistPtPzResidMotherFst_%i", MOTHER_PDG_IDS[iDecay]),
               "Pt-Pz Resid histo; p_{T}, GeV; Pz residual, GeV", LogBinsX, LogWidthX, 50, -0.15, 0.15);
    fHistPtPtResidMotherFst[iDecay] =
      new TH2F(Form("fHistPtPtResidMotherFst_%i", MOTHER_PDG_IDS[iDecay]),
               "Pt-Pt Resid histo; p_{T}, GeV; Pt residual, GeV", LogBinsX, LogWidthX, 50, -0.15, 0.15);
    fHistPtMassResidMotherFst[iDecay] =
      new TH2F(Form("fHistPtMassResidMotherFst_%i", MOTHER_PDG_IDS[iDecay]),
               "Pt-Mass Resid histo; p_{T}, GeV; Mass residual, GeV", LogBinsX, LogWidthX, 50, -0.4, 0.4);
    fHistPtEResidMotherFst[iDecay] =
      new TH2F(Form("fHistPtEResidMotherFst_%i", MOTHER_PDG_IDS[iDecay]),
               "Pt-E Resid histo; p_{T}, GeV; E residual, GeV", LogBinsX, LogWidthX, 50, -0.15, 0.15);
    fHistPtXResidMotherLst[iDecay] =
      new TH2F(Form("fHistPtXResidMotherLst_%i", MOTHER_PDG_IDS[iDecay]),
               "Pt-X Resid histo; p_{T}, GeV; X residual, cm", LogBinsX, LogWidthX, 100, -1e-1, 1e-1);
    fHistPtYResidMotherLst[iDecay] =
      new TH2F(Form("fHistPtYResidMotherLst_%i", MOTHER_PDG_IDS[iDecay]),
               "Pt-Y Resid histo; p_{T}, GeV; Y residual, cm", LogBinsX, LogWidthX, 100, -1e-1, 1e-1);
    fHistPtZResidMotherLst[iDecay] =
      new TH2F(Form("fHistPtZResidMotherLst_%i", MOTHER_PDG_IDS[iDecay]),
               "Pt-Z Resid histo; p_{T}, GeV; Z residual, cm", LogBinsX, LogWidthX, 100, -1e-1, 1e-1);
    fHistPtPxResidMotherLst[iDecay] =
      new TH2F(Form("fHistPtPxResidMotherLst_%i", MOTHER_PDG_IDS[iDecay]),
               "Pt-Px Resid histo; p_{T}, GeV; Px residual, GeV", LogBinsX, LogWidthX, 50, -0.15, 0.15);
    fHistPtPyResidMotherLst[iDecay] =
      new TH2F(Form("fHistPtPyResidMotherLst_%i", MOTHER_PDG_IDS[iDecay]),
               "Pt-Py Resid histo; p_{T}, GeV; Py residual, GeV", LogBinsX, LogWidthX, 50, -0.15, 0.15);
    fHistPtPzResidMotherLst[iDecay] =
      new TH2F(Form("fHistPtPzResidMotherLst_%i", MOTHER_PDG_IDS[iDecay]),
               "Pt-Pz Resid histo; p_{T}, GeV; Pz residual, GeV", LogBinsX, LogWidthX, 50, -0.15, 0.15);
    fHistPtPtResidMotherLst[iDecay] =
      new TH2F(Form("fHistPtPtResidMotherLst_%i", MOTHER_PDG_IDS[iDecay]),
               "Pt-Pt Resid histo; p_{T}, GeV; Pt residual, GeV", LogBinsX, LogWidthX, 50, -0.15, 0.15);
    fHistPtMassResidMotherLst[iDecay] =
      new TH2F(Form("fHistPtMassResidMotherLst_%i", MOTHER_PDG_IDS[iDecay]),
               "Pt-Mass Resid histo; p_{T}, GeV; Mass residual, GeV", LogBinsX, LogWidthX, 50, -0.4, 0.4);
    fHistPtEResidMotherLst[iDecay] =
      new TH2F(Form("fHistPtEResidMotherLst_%i", MOTHER_PDG_IDS[iDecay]),
               "Pt-E Resid histo; p_{T}, GeV; E residual, GeV", LogBinsX, LogWidthX, 50, -0.15, 0.15);

    for (int i = 0; i < NUM_OF_DAUGHTERS[iDecay]; i++) {
      fHistPtXResidDaughter[iDecay][i] =
        new TH2F(Form("fHistPtXResidDaughter%i_%i", MOTHER_PDG_IDS[iDecay], i + 1),
                 "Pt-X Resid histo; p_{T}, GeV; X residual, cm", LogBinsX, LogWidthX, 100, -1e-1, 1e-1);
      fHistPtYResidDaughter[iDecay][i] =
        new TH2F(Form("fHistPtYResidDaughter%i_%i", MOTHER_PDG_IDS[iDecay], i + 1),
                 "Pt-Y Resid histo; p_{T}, GeV; Y residual, cm", LogBinsX, LogWidthX, 100, -1e-1, 1e-1);
      fHistPtZResidDaughter[iDecay][i] =
        new TH2F(Form("fHistPtZResidDaughter%i_%i", MOTHER_PDG_IDS[iDecay], i + 1),
                 "Pt-Z Resid histo; p_{T}, GeV; Z residual, cm", LogBinsX, LogWidthX, 100, -1e-1, 1e-1);
      fHistPtPxResidDaughter[iDecay][i] =
        new TH2F(Form("fHistPtPxResidDaughter%i_%i", MOTHER_PDG_IDS[iDecay], i + 1),
                 "Pt-Px Resid histo; p_{T}, GeV; Px residual, GeV", LogBinsX, LogWidthX, 50, -0.15, 0.15);
      fHistPtPyResidDaughter[iDecay][i] =
        new TH2F(Form("fHistPtPyResidDaughter%i_%i", MOTHER_PDG_IDS[iDecay], i + 1),
                 "Pt-Py Resid histo; p_{T}, GeV; Py residual, GeV", LogBinsX, LogWidthX, 50, -0.15, 0.15);
      fHistPtPzResidDaughter[iDecay][i] =
        new TH2F(Form("fHistPtPzResidDaughter%i_%i", MOTHER_PDG_IDS[iDecay], i + 1),
                 "Pt-Pz Resid histo; p_{T}, GeV; Pz residual, GeV", LogBinsX, LogWidthX, 50, -0.15, 0.15);
      fHistPtPtResidDaughter[iDecay][i] =
        new TH2F(Form("fHistPtPtResidDaughter%i_%i", MOTHER_PDG_IDS[iDecay], i + 1),
                 "Pt-Pt Resid histo; p_{T}, GeV; Pt residual, GeV", LogBinsX, LogWidthX, 50, -0.15, 0.15);
      fHistPtMassResidDaughter[iDecay][i] =
        new TH2F(Form("fHistPtMassResidDaughter%i_%i", MOTHER_PDG_IDS[iDecay], i + 1),
                 "Pt-Mass Resid histo; p_{T}, GeV; Mass residual, GeV", LogBinsX, LogWidthX, 50, -0.4, 0.4);
      fHistPtEResidDaughter[iDecay][i] =
        new TH2F(Form("fHistPtEResidMother%i_%i", MOTHER_PDG_IDS[iDecay], i + 1),
                 "Pt-E Resid histo; p_{T}, GeV; E residual, GeV", LogBinsX, LogWidthX, 50, -0.15, 0.15);
    }
    //
    fHistXResidMotherFst[iDecay]    = new TH1D(Form("fHistXResidMotherFst_%i", MOTHER_PDG_IDS[iDecay]),
                                               "X Resid histo; X residual, cm; counts", 100, -1e-1, 1e-1);
    fHistYResidMotherFst[iDecay]    = new TH1D(Form("fHistYResidMotherFst_%i", MOTHER_PDG_IDS[iDecay]),
                                               "Y Resid histo; Y residual, cm; counts", 100, -1e-1, 1e-1);
    fHistZResidMotherFst[iDecay]    = new TH1D(Form("fHistZResidMotherFst_%i", MOTHER_PDG_IDS[iDecay]),
                                               "Z Resid histo; Z residual, cm; counts", 100, -1e-1, 1e-1);
    fHistPxResidMotherFst[iDecay]   = new TH1D(Form("fHistPxResidMotherFst_%i", MOTHER_PDG_IDS[iDecay]),
                                               "Px Resid histo; Px residual, GeV; counts", 50, -0.15, 0.15);
    fHistPyResidMotherFst[iDecay]   = new TH1D(Form("fHistPyResidMotherFst_%i", MOTHER_PDG_IDS[iDecay]),
                                               "Py Resid histo; Py residual, GeV; counts", 50, -0.15, 0.15);
    fHistPzResidMotherFst[iDecay]   = new TH1D(Form("fHistPzResidMotherFst_%i", MOTHER_PDG_IDS[iDecay]),
                                               "Pz Resid histo; Pz residual, GeV; counts", 50, -0.15, 0.15);
    fHistPtResidMotherFst[iDecay]   = new TH1D(Form("fHistPtResidMotherFst_%i", MOTHER_PDG_IDS[iDecay]),
                                               "Pt - Resid histo; Pt residual, GeV; counts", 50, -0.15, 0.15);
    fHistMassResidMotherFst[iDecay] = new TH1D(Form("fHistMassResidMotherFst_%i", MOTHER_PDG_IDS[iDecay]),
                                               "Mass Resid histo; Mass residual, GeV; counts", 50, -0.4, 0.4);
    fHistEResidMotherFst[iDecay]    = new TH1D(Form("fHistEResidMotherFst_%i", MOTHER_PDG_IDS[iDecay]),
                                               "E Resid histo; E residual, GeV; counts", 50, -0.15, 0.15);

    fHistXResidMotherLst[iDecay]    = new TH1D(Form("fHistXResidMotherLst_%i", MOTHER_PDG_IDS[iDecay]),
                                               "X Resid histo; X residual, cm; counts", 100, -1e-1, 1e-1);
    fHistYResidMotherLst[iDecay]    = new TH1D(Form("fHistYResidMotherLst_%i", MOTHER_PDG_IDS[iDecay]),
                                               "Y Resid histo; Y residual, cm; counts", 100, -1e-1, 1e-1);
    fHistZResidMotherLst[iDecay]    = new TH1D(Form("fHistZResidMotherLst_%i", MOTHER_PDG_IDS[iDecay]),
                                               "Z Resid histo; Z residual, cm; counts", 100, -1e-1, 1e-1);
    fHistPxResidMotherLst[iDecay]   = new TH1D(Form("fHistPxResidMotherLst_%i", MOTHER_PDG_IDS[iDecay]),
                                               "Px Resid histo; Px residual, GeV; counts", 50, -0.15, 0.15);
    fHistPyResidMotherLst[iDecay]   = new TH1D(Form("fHistPyResidMotherLst_%i", MOTHER_PDG_IDS[iDecay]),
                                               "Py Resid histo; Py residual, GeV; counts", 50, -0.15, 0.15);
    fHistPzResidMotherLst[iDecay]   = new TH1D(Form("fHistPzResidMotherLst_%i", MOTHER_PDG_IDS[iDecay]),
                                               "Pz Resid histo; Pz residual, GeV; counts", 50, -0.15, 0.15);
    fHistPtResidMotherLst[iDecay]   = new TH1D(Form("fHistPtResidMotherLst_%i", MOTHER_PDG_IDS[iDecay]),
                                               "Pt - Resid histo; Pt residual, GeV; counts", 50, -0.15, 0.15);
    fHistMassResidMotherLst[iDecay] = new TH1D(Form("fHistMassResidMotherLst_%i", MOTHER_PDG_IDS[iDecay]),
                                               "Mass Resid histo; Mass residual, GeV; counts", 50, -0.4, 0.4);
    fHistEResidMotherLst[iDecay]    = new TH1D(Form("fHistEResidMotherLst_%i", MOTHER_PDG_IDS[iDecay]),
                                               "E Resid histo; E residual, GeV; counts", 50, -0.15, 0.15);

    for (int i = 0; i < NUM_OF_DAUGHTERS[iDecay]; i++) {
      fHistXResidDaughter[iDecay][i]    = new TH1D(Form("fHistXResidDaughter%i_%i", MOTHER_PDG_IDS[iDecay], i),
                                                   "X Resid histo; X residual, cm; counts", 100, -1e-1, 1e-1);
      fHistYResidDaughter[iDecay][i]    = new TH1D(Form("fHistYResidDaughter%i_%i", MOTHER_PDG_IDS[iDecay], i),
                                                   "Y Resid histo; Y residual, cm; counts", 100, -1e-1, 1e-1);
      fHistZResidDaughter[iDecay][i]    = new TH1D(Form("fHistZResidDaughter%i_%i", MOTHER_PDG_IDS[iDecay], i),
                                                   "Z Resid histo; Z residual, cm; counts", 100, -1e-1, 1e-1);
      fHistPxResidDaughter[iDecay][i]   = new TH1D(Form("fHistPxResidDaughter%i_%i", MOTHER_PDG_IDS[iDecay], i),
                                                   "Px Resid histo; Px residual, GeV; counts", 50, -0.15, 0.15);
      fHistPyResidDaughter[iDecay][i]   = new TH1D(Form("fHistPyResidDaughter%i_%i", MOTHER_PDG_IDS[iDecay], i),
                                                   "Py Resid histo; Py residual, GeV; counts", 50, -0.15, 0.15);
      fHistPzResidDaughter[iDecay][i]   = new TH1D(Form("fHistPzResidDaughter%i_%i", MOTHER_PDG_IDS[iDecay], i),
                                                   "Pz Resid histo; Pz residual, GeV; counts", 50, -0.15, 0.15);
      fHistPtResidDaughter[iDecay][i]   = new TH1D(Form("fHistPtResidDaughter%i_%i", MOTHER_PDG_IDS[iDecay], i),
                                                   "Pt Resid histo;  residual, GeV; counts", 50, -0.15, 0.15);
      fHistMassResidDaughter[iDecay][i] = new TH1D(Form("fHistMassResidDaughter%i_%i", MOTHER_PDG_IDS[iDecay], i),
                                                   "Mass Resid histo; Mass residual, GeV; counts", 50, -0.4, 0.4);
      fHistEResidDaughter[iDecay][i]    = new TH1D(Form("fHistEResidDaughter%i_%i", MOTHER_PDG_IDS[iDecay], i),
                                                   "E Resid histo; E residual, GeV; counts", 50, -0.15, 0.15);
    }
    //
    fGraphPtXMeanResidMother[iDecay]    = new TGraphErrors();
    fGraphPtYMeanResidMother[iDecay]    = new TGraphErrors();
    fGraphPtZMeanResidMother[iDecay]    = new TGraphErrors();
    fGraphPtPxMeanResidMother[iDecay]   = new TGraphErrors();
    fGraphPtPyMeanResidMother[iDecay]   = new TGraphErrors();
    fGraphPtPzMeanResidMother[iDecay]   = new TGraphErrors();
    fGraphPtPtMeanResidMother[iDecay]   = new TGraphErrors();
    fGraphPtMassMeanResidMother[iDecay] = new TGraphErrors();
    fGraphPtEMeanResidMother[iDecay]    = new TGraphErrors();

    for (int i = 0; i < NUM_OF_DAUGHTERS[iDecay]; i++) {
      fGraphPtXMeanResidDaughter[iDecay][i]    = new TGraphErrors();
      fGraphPtYMeanResidDaughter[iDecay][i]    = new TGraphErrors();
      fGraphPtZMeanResidDaughter[iDecay][i]    = new TGraphErrors();
      fGraphPtPxMeanResidDaughter[iDecay][i]   = new TGraphErrors();
      fGraphPtPyMeanResidDaughter[iDecay][i]   = new TGraphErrors();
      fGraphPtPzMeanResidDaughter[iDecay][i]   = new TGraphErrors();
      fGraphPtPtMeanResidDaughter[iDecay][i]   = new TGraphErrors();
      fGraphPtMassMeanResidDaughter[iDecay][i] = new TGraphErrors();
      fGraphPtEMeanResidDaughter[iDecay][i]    = new TGraphErrors();
    }
    //
    fGraphPtXResolMother[iDecay]    = new TGraphErrors();
    fGraphPtYResolMother[iDecay]    = new TGraphErrors();
    fGraphPtZResolMother[iDecay]    = new TGraphErrors();
    fGraphPtPxResolMother[iDecay]   = new TGraphErrors();
    fGraphPtPyResolMother[iDecay]   = new TGraphErrors();
    fGraphPtPzResolMother[iDecay]   = new TGraphErrors();
    fGraphPtPtResolMother[iDecay]   = new TGraphErrors();
    fGraphPtMassResolMother[iDecay] = new TGraphErrors();
    fGraphPtEResolMother[iDecay]    = new TGraphErrors();
    for (int i = 0; i < NUM_OF_DAUGHTERS[iDecay]; i++) {
      fGraphPtXResolDaughter[iDecay][i]    = new TGraphErrors();
      fGraphPtYResolDaughter[iDecay][i]    = new TGraphErrors();
      fGraphPtZResolDaughter[iDecay][i]    = new TGraphErrors();
      fGraphPtPxResolDaughter[iDecay][i]   = new TGraphErrors();
      fGraphPtPyResolDaughter[iDecay][i]   = new TGraphErrors();
      fGraphPtPzResolDaughter[iDecay][i]   = new TGraphErrors();
      fGraphPtPtResolDaughter[iDecay][i]   = new TGraphErrors();
      fGraphPtMassResolDaughter[iDecay][i] = new TGraphErrors();
      fGraphPtEResolDaughter[iDecay][i]    = new TGraphErrors();
    }
  }
}

void ResidPtDepHistos::Fill(int iDecay, const MCParticleStruct& mcPart)
{
  //....
}
void ResidPtDepHistos::Fill(int iDecay, const KFParticleStruct& kfPart)
{
  //....
}

// void ResidPtDepHistos::Fill(const MCParticleStruct& mcPart, const
// KFParticleStruct& kfPartDecay, const KFParticleStruct& kfPartProduction){};

void ResidPtDepHistos::Fill(int iDecay, const MCParticleStruct& mcPart, const KFParticleStruct& kfPart) {};

void ResidPtDepHistos::Fill(int iDecay, const MCParticleStruct& mcPart, const KFParticleStruct& kfPartDecay,
                            const KFParticleStruct& kfPartProduction)
{
  if (kfPartDecay.isMother) {
    // setup mother particle from KFAR data and fill some histos
    float x_resid_    = mcPart.finalX - kfPartDecay.x;
    float y_resid_    = mcPart.finalY - kfPartDecay.y;
    float z_resid_    = mcPart.finalZ - kfPartDecay.z;
    float px_resid_   = mcPart.finalPX - kfPartDecay.pX;
    float py_resid_   = mcPart.finalPY - kfPartDecay.pY;
    float pz_resid_   = mcPart.finalPZ - kfPartDecay.pZ;
    float pt_resid_   = mcPart.finalPT() - kfPartDecay.pT;
    float mass_resid_ = mcPart.mass - kfPartDecay.mass;
    float e_resid_    = mcPart.finalE() - kfPartDecay.e;

    fHistPtXResidMotherLst[iDecay]->Fill(mcPart.finalPT(), x_resid_);
    fHistPtYResidMotherLst[iDecay]->Fill(mcPart.finalPT(), y_resid_);
    fHistPtZResidMotherLst[iDecay]->Fill(mcPart.finalPT(), z_resid_);
    fHistPtPxResidMotherLst[iDecay]->Fill(mcPart.finalPT(), px_resid_);
    fHistPtPyResidMotherLst[iDecay]->Fill(mcPart.finalPT(), py_resid_);
    fHistPtPzResidMotherLst[iDecay]->Fill(mcPart.finalPT(), pz_resid_);
    fHistPtPtResidMotherLst[iDecay]->Fill(mcPart.finalPT(), pt_resid_);
    fHistPtMassResidMotherLst[iDecay]->Fill(mcPart.finalPT(), mass_resid_);
    fHistPtEResidMotherLst[iDecay]->Fill(mcPart.finalPT(), e_resid_);

    // setup mother particle from KFAR data and fill some histos
    float x_resid_prod_    = mcPart.initialX - kfPartProduction.x;
    float y_resid_prod_    = mcPart.initialY - kfPartProduction.y;
    float z_resid_prod_    = mcPart.initialZ - kfPartProduction.z;
    float px_resid_prod_   = mcPart.initialPX - kfPartProduction.pX;
    float py_resid_prod_   = mcPart.initialPY - kfPartProduction.pY;
    float pz_resid_prod_   = mcPart.initialPZ - kfPartProduction.pZ;
    float pt_resid_prod_   = mcPart.initialPT() - kfPartProduction.pT;
    float mass_resid_prod_ = mcPart.mass - kfPartProduction.mass;
    float e_resid_prod_    = mcPart.initialE() - kfPartProduction.e;

    fHistPtXResidMotherFst[iDecay]->Fill(mcPart.initialPT(), x_resid_prod_);
    fHistPtYResidMotherFst[iDecay]->Fill(mcPart.initialPT(), y_resid_prod_);
    fHistPtZResidMotherFst[iDecay]->Fill(mcPart.initialPT(), z_resid_prod_);
    fHistPtPxResidMotherFst[iDecay]->Fill(mcPart.initialPT(), px_resid_prod_);
    fHistPtPyResidMotherFst[iDecay]->Fill(mcPart.initialPT(), py_resid_prod_);
    fHistPtPzResidMotherFst[iDecay]->Fill(mcPart.initialPT(), pz_resid_prod_);
    fHistPtPtResidMotherFst[iDecay]->Fill(mcPart.initialPT(), pt_resid_prod_);
    fHistPtMassResidMotherFst[iDecay]->Fill(mcPart.initialPT(), mass_resid_prod_);
    fHistPtEResidMotherFst[iDecay]->Fill(mcPart.initialPT(), e_resid_prod_);
  }
  if (kfPartDecay.isDaughter) {
    int daughterNum = kfPartDecay.id - 1;  // is it correct???

    float x_resid_    = mcPart.initialX - kfPartDecay.x;
    float y_resid_    = mcPart.initialY - kfPartDecay.y;
    float z_resid_    = mcPart.initialZ - kfPartDecay.z;
    float px_resid_   = mcPart.initialPX - kfPartDecay.pX;
    float py_resid_   = mcPart.initialPY - kfPartDecay.pY;
    float pz_resid_   = mcPart.initialPZ - kfPartDecay.pZ;
    float pt_resid_   = mcPart.initialPT() - kfPartDecay.pT;
    float mass_resid_ = mcPart.mass - kfPartDecay.mass;
    float e_resid_    = mcPart.initialE() - kfPartDecay.e;

    fHistPtXResidDaughter[iDecay][daughterNum]->Fill(mcPart.initialPT(), x_resid_);
    fHistPtYResidDaughter[iDecay][daughterNum]->Fill(mcPart.initialPT(), y_resid_);
    fHistPtZResidDaughter[iDecay][daughterNum]->Fill(mcPart.initialPT(), z_resid_);
    fHistPtPxResidDaughter[iDecay][daughterNum]->Fill(mcPart.initialPT(), px_resid_);
    fHistPtPyResidDaughter[iDecay][daughterNum]->Fill(mcPart.initialPT(), py_resid_);
    fHistPtPzResidDaughter[iDecay][daughterNum]->Fill(mcPart.initialPT(), pz_resid_);
    fHistPtPtResidDaughter[iDecay][daughterNum]->Fill(mcPart.initialPT(), pt_resid_);
    fHistPtMassResidDaughter[iDecay][daughterNum]->Fill(mcPart.initialPT(), mass_resid_);
    fHistPtEResidDaughter[iDecay][daughterNum]->Fill(mcPart.initialPT(), e_resid_);
  }
}

void ResidPtDepHistos::PreWriteProcess()
{
  for (int iDecay = 0; iDecay < NUM_OF_DECAYS; iDecay++) {
    fHistXResidMotherFst[iDecay]    = fHistPtXResidMotherFst[iDecay]->ProjectionY();
    fHistYResidMotherFst[iDecay]    = fHistPtYResidMotherFst[iDecay]->ProjectionY();
    fHistZResidMotherFst[iDecay]    = fHistPtZResidMotherFst[iDecay]->ProjectionY();
    fHistPxResidMotherFst[iDecay]   = fHistPtPxResidMotherFst[iDecay]->ProjectionY();
    fHistPyResidMotherFst[iDecay]   = fHistPtPyResidMotherFst[iDecay]->ProjectionY();
    fHistPzResidMotherFst[iDecay]   = fHistPtPzResidMotherFst[iDecay]->ProjectionY();
    fHistPtResidMotherFst[iDecay]   = fHistPtPtResidMotherFst[iDecay]->ProjectionY();
    fHistMassResidMotherFst[iDecay] = fHistPtMassResidMotherFst[iDecay]->ProjectionY();
    fHistEResidMotherFst[iDecay]    = fHistPtEResidMotherFst[iDecay]->ProjectionY();

    fHistXResidMotherLst[iDecay]    = fHistPtXResidMotherLst[iDecay]->ProjectionY();
    fHistYResidMotherLst[iDecay]    = fHistPtYResidMotherLst[iDecay]->ProjectionY();
    fHistZResidMotherLst[iDecay]    = fHistPtZResidMotherLst[iDecay]->ProjectionY();
    fHistPxResidMotherLst[iDecay]   = fHistPtPxResidMotherLst[iDecay]->ProjectionY();
    fHistPyResidMotherLst[iDecay]   = fHistPtPyResidMotherLst[iDecay]->ProjectionY();
    fHistPzResidMotherLst[iDecay]   = fHistPtPzResidMotherLst[iDecay]->ProjectionY();
    fHistPtResidMotherLst[iDecay]   = fHistPtPtResidMotherLst[iDecay]->ProjectionY();
    fHistMassResidMotherLst[iDecay] = fHistPtMassResidMotherLst[iDecay]->ProjectionY();
    fHistEResidMotherLst[iDecay]    = fHistPtEResidMotherLst[iDecay]->ProjectionY();

    for (int i = 0; i < NUM_OF_DAUGHTERS[iDecay]; i++) {
      fHistXResidDaughter[iDecay][i]    = fHistPtXResidDaughter[iDecay][i]->ProjectionY();
      fHistYResidDaughter[iDecay][i]    = fHistPtYResidDaughter[iDecay][i]->ProjectionY();
      fHistZResidDaughter[iDecay][i]    = fHistPtZResidDaughter[iDecay][i]->ProjectionY();
      fHistPxResidDaughter[iDecay][i]   = fHistPtPxResidDaughter[iDecay][i]->ProjectionY();
      fHistPyResidDaughter[iDecay][i]   = fHistPtPyResidDaughter[iDecay][i]->ProjectionY();
      fHistPzResidDaughter[iDecay][i]   = fHistPtPzResidDaughter[iDecay][i]->ProjectionY();
      fHistPtResidDaughter[iDecay][i]   = fHistPtPtResidDaughter[iDecay][i]->ProjectionY();
      fHistMassResidDaughter[iDecay][i] = fHistPtMassResidDaughter[iDecay][i]->ProjectionY();
      fHistEResidDaughter[iDecay][i]    = fHistPtEResidDaughter[iDecay][i]->ProjectionY();
    }
    ///////
    fGraphPtXMeanResidMother[iDecay] = MakeMeanGraphFromTH2(fHistPtXResidMotherLst[iDecay]);
    fGraphPtXMeanResidMother[iDecay]->SetNameTitle(Form("X Mean Resid_%i", MOTHER_PDG_IDS[iDecay]),
                                                   "XMeanResid; p_{T}, GeV; X, cm");
    fGraphPtYMeanResidMother[iDecay] = MakeMeanGraphFromTH2(fHistPtYResidMotherLst[iDecay]);
    fGraphPtYMeanResidMother[iDecay]->SetNameTitle(Form("Y Mean Resid_%i", MOTHER_PDG_IDS[iDecay]),
                                                   "YMeanResid; p_{T}, GeV; Y, cm");
    fGraphPtZMeanResidMother[iDecay] = MakeMeanGraphFromTH2(fHistPtZResidMotherLst[iDecay]);
    fGraphPtZMeanResidMother[iDecay]->SetNameTitle(Form("Z Mean Resid_%i", MOTHER_PDG_IDS[iDecay]),
                                                   "ZMeanResid; p_{T}, GeV; Z, cm");
    fGraphPtPxMeanResidMother[iDecay] = MakeMeanGraphFromTH2(fHistPtPxResidMotherLst[iDecay]);
    fGraphPtPxMeanResidMother[iDecay]->SetNameTitle(Form("Px Mean Resid_%i", MOTHER_PDG_IDS[iDecay]),
                                                    "PxMeanResid; p_{T}, GeV; Px, GeV");
    fGraphPtPyMeanResidMother[iDecay] = MakeMeanGraphFromTH2(fHistPtPyResidMotherLst[iDecay]);
    fGraphPtPyMeanResidMother[iDecay]->SetNameTitle(Form("Py Mean Resid_%i", MOTHER_PDG_IDS[iDecay]),
                                                    "PyMeanResid; p_{T}, GeV; Py, GeV");
    fGraphPtPzMeanResidMother[iDecay] = MakeMeanGraphFromTH2(fHistPtPzResidMotherLst[iDecay]);
    fGraphPtPzMeanResidMother[iDecay]->SetNameTitle(Form("Pz Mean Resid_%i", MOTHER_PDG_IDS[iDecay]),
                                                    "PzMeanResid; p_{T}, GeV; Pz, GeV");
    fGraphPtPtMeanResidMother[iDecay] = MakeMeanGraphFromTH2(fHistPtPtResidMotherLst[iDecay]);
    fGraphPtPtMeanResidMother[iDecay]->SetNameTitle(Form("Pt Mean Resid_%i", MOTHER_PDG_IDS[iDecay]),
                                                    "PtMeanResid; p_{T}, GeV; Pt, GeV");
    fGraphPtMassMeanResidMother[iDecay] = MakeMeanGraphFromTH2(fHistPtMassResidMotherLst[iDecay]);
    fGraphPtMassMeanResidMother[iDecay]->SetNameTitle(Form("Mass Mean Resid_%i", MOTHER_PDG_IDS[iDecay]),
                                                      "MassMeanResid; p_{T}, GeV; Mass, GeV");
    fGraphPtEMeanResidMother[iDecay] = MakeMeanGraphFromTH2(fHistPtEResidMotherLst[iDecay]);
    fGraphPtEMeanResidMother[iDecay]->SetNameTitle(Form("E Mean Resid_%i", MOTHER_PDG_IDS[iDecay]),
                                                   "EMeanResid; p_{T}, GeV; E, GeV");

    for (int i = 0; i < NUM_OF_DAUGHTERS[iDecay]; i++) {
      fGraphPtXMeanResidDaughter[iDecay][i] = MakeMeanGraphFromTH2(fHistPtXResidDaughter[iDecay][i]);
      fGraphPtXMeanResidDaughter[iDecay][i]->SetNameTitle(Form("X Mean Resid_%i", MOTHER_PDG_IDS[iDecay]),
                                                          "XMeanResid; p_{T}, GeV; X, cm");
      fGraphPtYMeanResidDaughter[iDecay][i] = MakeMeanGraphFromTH2(fHistPtYResidDaughter[iDecay][i]);
      fGraphPtYMeanResidDaughter[iDecay][i]->SetNameTitle(Form("Y Mean Resid_%i", MOTHER_PDG_IDS[iDecay]),
                                                          "YMeanResid; p_{T}, GeV; Y, cm");
      fGraphPtZMeanResidDaughter[iDecay][i] = MakeMeanGraphFromTH2(fHistPtZResidDaughter[iDecay][i]);
      fGraphPtZMeanResidDaughter[iDecay][i]->SetNameTitle(Form("Z Mean Resid_%i", MOTHER_PDG_IDS[iDecay]),
                                                          "ZMeanResid; p_{T}, GeV; Z, cm");
      fGraphPtPxMeanResidDaughter[iDecay][i] = MakeMeanGraphFromTH2(fHistPtPxResidDaughter[iDecay][i]);
      fGraphPtPxMeanResidDaughter[iDecay][i]->SetNameTitle(Form("Px Mean Resid_%i", MOTHER_PDG_IDS[iDecay]),
                                                           "PxMeanResid; p_{T}, GeV; Px, GeV");
      fGraphPtPyMeanResidDaughter[iDecay][i] = MakeMeanGraphFromTH2(fHistPtPyResidDaughter[iDecay][i]);
      fGraphPtPyMeanResidDaughter[iDecay][i]->SetNameTitle(Form("Py Mean Resid_%i", MOTHER_PDG_IDS[iDecay]),
                                                           "PyMeanResid; p_{T}, GeV; Py, GeV");
      fGraphPtPzMeanResidDaughter[iDecay][i] = MakeMeanGraphFromTH2(fHistPtPzResidDaughter[iDecay][i]);
      fGraphPtPzMeanResidDaughter[iDecay][i]->SetNameTitle(Form("Pz Mean Resid_%i", MOTHER_PDG_IDS[iDecay]),
                                                           "PzMeanResid; p_{T}, GeV; Pz, GeV");
      fGraphPtPtMeanResidDaughter[iDecay][i] = MakeMeanGraphFromTH2(fHistPtPtResidDaughter[iDecay][i]);
      fGraphPtPtMeanResidDaughter[iDecay][i]->SetNameTitle(Form("Pt Mean Resid_%i", MOTHER_PDG_IDS[iDecay]),
                                                           "PtMeanResid; p_{T}, GeV; Pt, GeV");
      fGraphPtMassMeanResidDaughter[iDecay][i] = MakeMeanGraphFromTH2(fHistPtMassResidDaughter[iDecay][i]);
      fGraphPtMassMeanResidDaughter[iDecay][i]->SetNameTitle(Form("Mass Mean Resid_%i", MOTHER_PDG_IDS[iDecay]),
                                                             "MassMeanResid; p_{T}, GeV; Mass, GeV");
      fGraphPtEMeanResidDaughter[iDecay][i] = MakeMeanGraphFromTH2(fHistPtEResidDaughter[iDecay][i]);
      fGraphPtEMeanResidDaughter[iDecay][i]->SetNameTitle(Form("E Mean Resid_%i", MOTHER_PDG_IDS[iDecay]),
                                                          "EMeanResid; p_{T}, GeV; E, GeV");
    }
    ///////
    fGraphPtXResolMother[iDecay] = MakeSigmaGraphFromTH2(fHistPtXResidMotherLst[iDecay]);
    fGraphPtXResolMother[iDecay]->SetNameTitle(Form("X Resid Width_%i", MOTHER_PDG_IDS[iDecay]),
                                               "XResol; p_{T}, GeV; X resolution, cm");
    fGraphPtYResolMother[iDecay] = MakeSigmaGraphFromTH2(fHistPtYResidMotherLst[iDecay]);
    fGraphPtYResolMother[iDecay]->SetNameTitle(Form("Y Resid Width_%i", MOTHER_PDG_IDS[iDecay]),
                                               "YResol; p_{T}, GeV; Y resolution, cm");
    fGraphPtZResolMother[iDecay] = MakeSigmaGraphFromTH2(fHistPtZResidMotherLst[iDecay]);
    fGraphPtZResolMother[iDecay]->SetNameTitle(Form("Z Resid Width_%i", MOTHER_PDG_IDS[iDecay]),
                                               "ZResol; p_{T}, GeV; Z resolution, cm");
    fGraphPtPxResolMother[iDecay] = MakeSigmaGraphFromTH2(fHistPtPxResidMotherLst[iDecay]);
    fGraphPtPxResolMother[iDecay]->SetNameTitle(Form("Px Resid Width_%i", MOTHER_PDG_IDS[iDecay]),
                                                "PxResol; p_{T}, GeV; Px resolution, GeV");
    fGraphPtPyResolMother[iDecay] = MakeSigmaGraphFromTH2(fHistPtPyResidMotherLst[iDecay]);
    fGraphPtPyResolMother[iDecay]->SetNameTitle(Form("Py Resid Width_%i", MOTHER_PDG_IDS[iDecay]),
                                                "PyResol; p_{T}, GeV; Py resolution, GeV");
    fGraphPtPzResolMother[iDecay] = MakeSigmaGraphFromTH2(fHistPtPzResidMotherLst[iDecay]);
    fGraphPtPzResolMother[iDecay]->SetNameTitle(Form("Pz Resid Width_%i", MOTHER_PDG_IDS[iDecay]),
                                                "PzResol; p_{T}, GeV; Pz resolution, GeV");
    fGraphPtPtResolMother[iDecay] = MakeSigmaGraphFromTH2(fHistPtPtResidMotherLst[iDecay]);
    fGraphPtPtResolMother[iDecay]->SetNameTitle(Form("Pt Resid Width_%i", MOTHER_PDG_IDS[iDecay]),
                                                "PtResol; p_{T}, GeV; Pt resolution, GeV");
    fGraphPtMassResolMother[iDecay] = MakeSigmaGraphFromTH2(fHistPtMassResidMotherLst[iDecay]);
    fGraphPtMassResolMother[iDecay]->SetNameTitle(Form("Mass Resid Width_%i", MOTHER_PDG_IDS[iDecay]),
                                                  "MassResol; p_{T}, GeV; Mass resolution, GeV");
    fGraphPtEResolMother[iDecay] = MakeSigmaGraphFromTH2(fHistPtEResidMotherLst[iDecay]);
    fGraphPtEResolMother[iDecay]->SetNameTitle(Form("E Resid Width_%i", MOTHER_PDG_IDS[iDecay]),
                                               "EResol; p_{T}, GeV; E resolution, GeV");

    for (int i = 0; i < NUM_OF_DAUGHTERS[iDecay]; i++) {
      fGraphPtXResolDaughter[iDecay][i] = MakeSigmaGraphFromTH2(fHistPtXResidDaughter[iDecay][i]);
      fGraphPtXResolDaughter[iDecay][i]->SetNameTitle(Form("X Resol_%i", MOTHER_PDG_IDS[iDecay]),
                                                      "XResol; p_{T}, GeV; X resolution, cm");
      fGraphPtYResolDaughter[iDecay][i] = MakeSigmaGraphFromTH2(fHistPtYResidDaughter[iDecay][i]);
      fGraphPtYResolDaughter[iDecay][i]->SetNameTitle(Form("Y Resol_%i", MOTHER_PDG_IDS[iDecay]),
                                                      "YResol; p_{T}, GeV; Y resolution, cm");
      fGraphPtZResolDaughter[iDecay][i] = MakeSigmaGraphFromTH2(fHistPtZResidDaughter[iDecay][i]);
      fGraphPtZResolDaughter[iDecay][i]->SetNameTitle(Form("Z Resol_%i", MOTHER_PDG_IDS[iDecay]),
                                                      "ZResol; p_{T}, GeV; Z resolution, cm");
      fGraphPtPxResolDaughter[iDecay][i] = MakeSigmaGraphFromTH2(fHistPtPxResidDaughter[iDecay][i]);
      fGraphPtPxResolDaughter[iDecay][i]->SetNameTitle(Form("Px Resol_%i", MOTHER_PDG_IDS[iDecay]),
                                                       "PxResol; p_{T}, GeV; Px resolution, GeV");
      fGraphPtPyResolDaughter[iDecay][i] = MakeSigmaGraphFromTH2(fHistPtPyResidDaughter[iDecay][i]);
      fGraphPtPyResolDaughter[iDecay][i]->SetNameTitle(Form("Py Resol_%i", MOTHER_PDG_IDS[iDecay]),
                                                       "PyResol; p_{T}, GeV; Py resolution, GeV");
      fGraphPtPzResolDaughter[iDecay][i] = MakeSigmaGraphFromTH2(fHistPtPzResidDaughter[iDecay][i]);
      fGraphPtPzResolDaughter[iDecay][i]->SetNameTitle(Form("Pz Resol_%i", MOTHER_PDG_IDS[iDecay]),
                                                       "PzResol; p_{T}, GeV; Pz resolution, GeV");
      fGraphPtPtResolDaughter[iDecay][i] = MakeSigmaGraphFromTH2(fHistPtPtResidDaughter[iDecay][i]);
      fGraphPtPtResolDaughter[iDecay][i]->SetNameTitle(Form("Pt Resol_%i", MOTHER_PDG_IDS[iDecay]),
                                                       "PtResol; p_{T}, GeV; Pt resolution, GeV");
      fGraphPtMassResolDaughter[iDecay][i] = MakeSigmaGraphFromTH2(fHistPtMassResidDaughter[iDecay][i]);
      fGraphPtMassResolDaughter[iDecay][i]->SetNameTitle(Form("Mass Resol_%i", MOTHER_PDG_IDS[iDecay]),
                                                         "MassResol; p_{T}, GeV; Mass resolution, GeV");
      fGraphPtEResolDaughter[iDecay][i] = MakeSigmaGraphFromTH2(fHistPtEResidDaughter[iDecay][i]);
      fGraphPtEResolDaughter[iDecay][i]->SetNameTitle(Form("E Resol_%i", MOTHER_PDG_IDS[iDecay]),
                                                      "EResol; p_{T}, GeV; E resolution, GeV");
    }
  }
}

bool ResidPtDepHistos::Check()
{
  cout << "Checking residuals..." << endl;
  bool Residsok = true;
  string errMsg = "";
  // TODO checks

  for (int iDecay = 0; iDecay < NUM_OF_DECAYS; iDecay++) {
    // check mean and width of Resids (x,y,z,px,py,pz,pt,m,e
    double xMean    = fHistXResidMotherLst[iDecay]->GetMean();
    double yMean    = fHistYResidMotherLst[iDecay]->GetMean();
    double zMean    = fHistZResidMotherLst[iDecay]->GetMean();
    double pxMean   = fHistPxResidMotherLst[iDecay]->GetMean();
    double pyMean   = fHistPyResidMotherLst[iDecay]->GetMean();
    double pzMean   = fHistPzResidMotherLst[iDecay]->GetMean();
    double ptMean   = fHistPtResidMotherLst[iDecay]->GetMean();
    double mMean    = fHistMassResidMotherLst[iDecay]->GetMean();
    double eMean    = fHistEResidMotherLst[iDecay]->GetMean();
    double xStdDev  = fHistXResidMotherLst[iDecay]->GetStdDev();
    double yStdDev  = fHistYResidMotherLst[iDecay]->GetStdDev();
    double zStdDev  = fHistZResidMotherLst[iDecay]->GetStdDev();
    double pxStdDev = fHistPxResidMotherLst[iDecay]->GetStdDev();
    double pyStdDev = fHistPyResidMotherLst[iDecay]->GetStdDev();
    double pzStdDev = fHistPzResidMotherLst[iDecay]->GetStdDev();
    double ptStdDev = fHistPtResidMotherLst[iDecay]->GetStdDev();
    double mStdDev  = fHistMassResidMotherLst[iDecay]->GetStdDev();
    double eStdDev  = fHistEResidMotherLst[iDecay]->GetStdDev();

    double epsilon = 0.1;

    double err_xMean    = epsilon;
    double err_yMean    = epsilon;
    double err_zMean    = epsilon;
    double err_pxMean   = epsilon;
    double err_pyMean   = epsilon;
    double err_pzMean   = epsilon;
    double err_ptMean   = epsilon;
    double err_mMean    = epsilon;
    double err_eMean    = epsilon;
    double err_xStdDev  = epsilon;
    double err_yStdDev  = epsilon;
    double err_zStdDev  = epsilon;
    double err_pxStdDev = epsilon;
    double err_pyStdDev = epsilon;
    double err_pzStdDev = epsilon;
    double err_ptStdDev = epsilon;
    double err_mStdDev  = epsilon;
    double err_eStdDev  = epsilon;

    if (fabs(xMean) > fabs(err_xMean)) {
      Residsok = false;
      errMsg += Form("\nERROR_RES: Decay %i: Resid x is biased: xMean = %F", MOTHER_PDG_IDS[iDecay], xMean);
    }
    if (fabs(yMean) > fabs(err_yMean)) {
      Residsok = false;
      errMsg += Form("\nERROR_RES: Decay %i: Resid y is biased: yMean = %F", MOTHER_PDG_IDS[iDecay], yMean);
    }
    if (fabs(zMean) > fabs(err_zMean)) {
      Residsok = false;
      errMsg += Form("\nERROR_RES: Decay %i: Resid z is biased: zMean = %F", MOTHER_PDG_IDS[iDecay], zMean);
    }
    if (fabs(pxMean) > fabs(err_pxMean)) {
      Residsok = false;
      errMsg += Form("\nERROR_RES: Decay %i: Resid px is biased: pxMean = %F", MOTHER_PDG_IDS[iDecay], pxMean);
    }
    if (fabs(pyMean) > fabs(err_pyMean)) {
      Residsok = false;
      errMsg += Form("\nERROR_RES: Decay %i: Resid py is biased: pyMean = %F", MOTHER_PDG_IDS[iDecay], pyMean);
    }
    if (fabs(pzMean) > fabs(err_pzMean)) {
      Residsok = false;
      errMsg += Form("\nERROR_RES: Decay %i: Resid pz is biased: pzMean = %F", MOTHER_PDG_IDS[iDecay], pzMean);
    }
    if (fabs(ptMean) > fabs(err_ptMean)) {
      Residsok = false;
      errMsg += Form("\nERROR_RES: Decay %i: Resid pT is biased: ptMean = %F", MOTHER_PDG_IDS[iDecay], ptMean);
    }
    if (fabs(mMean) > fabs(err_mMean)) {
      Residsok = false;
      errMsg += Form("\nERROR_RES: Decay %i: Resid m is biased: mMean = %F", MOTHER_PDG_IDS[iDecay], mMean);
    }
    if (fabs(zMean) > fabs(err_eMean)) {
      Residsok = false;
      errMsg += Form("\nERROR_RES: Decay %i: Resid E is biased: eMean = %F", MOTHER_PDG_IDS[iDecay], eMean);
    }

    // if(xStdDev>1+err_xStdDev) { Residsok = false; errMsg +=
    // Form("\nERROR_RES: Decay %i: Error x is underestimated: xStdDev =
    // %F",MOTHER_PDG_IDS[iDecay],xStdDev); } if(yStdDev>1+err_yStdDev) {
    // Residsok = false; errMsg += Form("\nERROR_RES: Decay %i: Error y is
    // underestimated: yStdDev = %F",MOTHER_PDG_IDS[iDecay],yStdDev); }
    // if(zStdDev>1+err_zStdDev) { Residsok = false; errMsg +=
    // Form("\nERROR_RES: Decay %i: Error z is underestimated: zStdDev =
    // %F",MOTHER_PDG_IDS[iDecay],zStdDev); } if(pxStdDev>1+err_pxStdDev) {
    // Residsok = false; errMsg += Form("\nERROR_RES: Decay %i: Error px is
    // underestimated: pxStdDev = %F",MOTHER_PDG_IDS[iDecay],pxStdDev); }
    // if(pyStdDev>1+err_pyStdDev) { Residsok = false; errMsg +=
    // Form("\nERROR_RES: Decay %i: Error py is underestimated: pyStdDev =
    // %F",MOTHER_PDG_IDS[iDecay],pyStdDev); } if(pzStdDev>1+err_pzStdDev) {
    // Residsok = false; errMsg += Form("\nERROR_RES: Decay %i: Error pz is
    // underestimated: pzStdDev = %F",MOTHER_PDG_IDS[iDecay],pzStdDev); }
    // if(ptStdDev>1+err_ptStdDev) { Residsok = false; errMsg +=
    // Form("\nERROR_RES: Decay %i: Error pT is underestimated: ptStdDev =
    // %F",MOTHER_PDG_IDS[iDecay],ptStdDev); } if(mStdDev>1+err_mStdDev) {
    // Residsok = false; errMsg += Form("\nERROR_RES: Decay %i: Error m is
    // underestimated: mStdDev = %F",MOTHER_PDG_IDS[iDecay],mStdDev); }
    // if(eStdDev>1+err_eStdDev) { Residsok = false; errMsg +=
    // Form("\nERROR_RES: Decay %i: Error E is underestimated: eStdDev =
    // %F",MOTHER_PDG_IDS[iDecay],eStdDev); }

    // if(xStdDev<1-err_xStdDev) { Residsok = false; errMsg +=
    // Form("\nERROR_RES: Decay %i: Error x is overestimated: xStdDev =
    // %F",MOTHER_PDG_IDS[iDecay],xStdDev); } if(yStdDev<1-err_yStdDev) {
    // Residsok = false; errMsg += Form("\nERROR_RES: Decay %i: Error y is
    // overestimated: yStdDev = %F",MOTHER_PDG_IDS[iDecay],yStdDev); }
    // if(zStdDev<1-err_zStdDev) { Residsok = false; errMsg +=
    // Form("\nERROR_RES: Decay %i: Error z is overestimated: zStdDev =
    // %F",MOTHER_PDG_IDS[iDecay],zStdDev); } if(pxStdDev<1-err_pxStdDev) {
    // Residsok = false; errMsg += Form("\nERROR_RES: Decay %i: Error px is
    // overestimated: pxStdDev = %F",MOTHER_PDG_IDS[iDecay],pxStdDev); }
    // if(pyStdDev<1-err_pyStdDev) { Residsok = false; errMsg +=
    // Form("\nERROR_RES: Decay %i: Error py is overestimated: pyStdDev =
    // %F",MOTHER_PDG_IDS[iDecay],pyStdDev); } if(pzStdDev<1-err_pzStdDev) {
    // Residsok = false; errMsg += Form("\nERROR_RES: Decay %i: Error pz is
    // overestimated: pzStdDev = %F",MOTHER_PDG_IDS[iDecay],pzStdDev); }
    // if(ptStdDev<1-err_ptStdDev) { Residsok = false; errMsg +=
    // Form("\nERROR_RES: Decay %i: Error pT is overestimated: ptStdDev =
    // %F",MOTHER_PDG_IDS[iDecay],ptStdDev); } if(mStdDev<1-err_mStdDev) {
    // Residsok = false; errMsg += Form("\nERROR_RES: Decay %i: Error m is
    // overestimated: mStdDev = %F",MOTHER_PDG_IDS[iDecay],mStdDev); }
    // if(eStdDev<1-err_eStdDev) { Residsok = false; errMsg +=
    // Form("\nERROR_RES: Decay %i: Error E is overestimated: eStdDev =
    // %F",MOTHER_PDG_IDS[iDecay],eStdDev); }
  }

  if (!Residsok) { cout << errMsg << endl; }

  return Residsok;
}

void ResidPtDepHistos::Write()
{
  TDirectory* Resid_Dir = outputFile->mkdir("ResidsVsPt");

  TDirectory* ResidPtDep_Dir = Resid_Dir->mkdir("ResidPtDependencies");
  std::array<TDirectory*, NUM_OF_DECAYS> DecayResidPtDep_Dir;
  std::array<TDirectory*, NUM_OF_DECAYS> MotherResidPtDep_Dir;
  std::array<std::array<TDirectory*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> DaughtersResidPtDep_Dir;
  for (int iDecay = 0; iDecay < NUM_OF_DECAYS; iDecay++) {
    DecayResidPtDep_Dir[iDecay]  = ResidPtDep_Dir->mkdir(Form("%i", MOTHER_PDG_IDS[iDecay]));
    MotherResidPtDep_Dir[iDecay] = DecayResidPtDep_Dir[iDecay]->mkdir(Form("Mother_%i", MOTHER_PDG_IDS[iDecay]));
    for (int i = 0; i < NUM_OF_DAUGHTERS[iDecay]; i++) {
      DaughtersResidPtDep_Dir[iDecay][i] = DecayResidPtDep_Dir[iDecay]->mkdir(
        Form("Daughter%i_%i->%i", i + 1, MOTHER_PDG_IDS[iDecay], DAUGHTER_PDG_IDS[iDecay][i]));
    }

    MotherResidPtDep_Dir[iDecay]->cd();

    fHistPtXResidMotherLst[iDecay]->Write();
    fHistPtYResidMotherLst[iDecay]->Write();
    fHistPtZResidMotherLst[iDecay]->Write();
    fHistPtPxResidMotherLst[iDecay]->Write();
    fHistPtPyResidMotherLst[iDecay]->Write();
    fHistPtPzResidMotherLst[iDecay]->Write();
    fHistPtPtResidMotherLst[iDecay]->Write();
    fHistPtMassResidMotherLst[iDecay]->Write();
    fHistPtEResidMotherLst[iDecay]->Write();

    fHistPtXResidMotherFst[iDecay]->Write();
    fHistPtYResidMotherFst[iDecay]->Write();
    fHistPtZResidMotherFst[iDecay]->Write();
    fHistPtPxResidMotherFst[iDecay]->Write();
    fHistPtPyResidMotherFst[iDecay]->Write();
    fHistPtPzResidMotherFst[iDecay]->Write();
    fHistPtPtResidMotherFst[iDecay]->Write();
    fHistPtMassResidMotherFst[iDecay]->Write();
    fHistPtEResidMotherFst[iDecay]->Write();

    for (int i = 0; i < NUM_OF_DAUGHTERS[iDecay]; i++) {
      DaughtersResidPtDep_Dir[iDecay][i]->cd();
      fHistPtXResidDaughter[iDecay][i]->Write();
      fHistPtYResidDaughter[iDecay][i]->Write();
      fHistPtZResidDaughter[iDecay][i]->Write();
      fHistPtPxResidDaughter[iDecay][i]->Write();
      fHistPtPyResidDaughter[iDecay][i]->Write();
      fHistPtPzResidDaughter[iDecay][i]->Write();
      fHistPtPtResidDaughter[iDecay][i]->Write();
      fHistPtMassResidDaughter[iDecay][i]->Write();
      fHistPtEResidDaughter[iDecay][i]->Write();
    }
  }
  /////////////////////////////////
  /////////////////////////////////

  TDirectory* ResidHistos_Dir = Resid_Dir->mkdir("ResidHistos");

  std::array<TDirectory*, NUM_OF_DECAYS> DecayResidHistos_Dir;
  std::array<TDirectory*, NUM_OF_DECAYS> MotherResidHistos_Dir;
  std::array<std::array<TDirectory*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> DaughtersResidHistos_Dir;
  for (int iDecay = 0; iDecay < NUM_OF_DECAYS; iDecay++) {
    DecayResidHistos_Dir[iDecay]  = ResidHistos_Dir->mkdir(Form("%i", MOTHER_PDG_IDS[iDecay]));
    MotherResidHistos_Dir[iDecay] = DecayResidHistos_Dir[iDecay]->mkdir(Form("Mother_%i", MOTHER_PDG_IDS[iDecay]));
    for (int i = 0; i < NUM_OF_DAUGHTERS[iDecay]; i++) {
      DaughtersResidHistos_Dir[iDecay][i] = DecayResidHistos_Dir[iDecay]->mkdir(
        Form("Daughter%i_%i->%i", i + 1, MOTHER_PDG_IDS[iDecay], DAUGHTER_PDG_IDS[iDecay][i]));
    }

    MotherResidHistos_Dir[iDecay]->cd();
    fHistXResidMotherLst[iDecay]->Write();
    fHistYResidMotherLst[iDecay]->Write();
    fHistZResidMotherLst[iDecay]->Write();
    fHistPxResidMotherLst[iDecay]->Write();
    fHistPyResidMotherLst[iDecay]->Write();
    fHistPzResidMotherLst[iDecay]->Write();
    fHistPtResidMotherLst[iDecay]->Write();
    fHistMassResidMotherLst[iDecay]->Write();
    fHistEResidMotherLst[iDecay]->Write();

    fHistXResidMotherFst[iDecay]->Write();
    fHistYResidMotherFst[iDecay]->Write();
    fHistZResidMotherFst[iDecay]->Write();
    fHistPxResidMotherFst[iDecay]->Write();
    fHistPyResidMotherFst[iDecay]->Write();
    fHistPzResidMotherFst[iDecay]->Write();
    fHistPtResidMotherFst[iDecay]->Write();
    fHistMassResidMotherFst[iDecay]->Write();
    fHistEResidMotherFst[iDecay]->Write();

    for (int i = 0; i < NUM_OF_DAUGHTERS[iDecay]; i++) {
      DaughtersResidHistos_Dir[iDecay][i]->cd();
      fHistXResidDaughter[iDecay][i]->Write();
      fHistYResidDaughter[iDecay][i]->Write();
      fHistZResidDaughter[iDecay][i]->Write();
      fHistPxResidDaughter[iDecay][i]->Write();
      fHistPyResidDaughter[iDecay][i]->Write();
      fHistPzResidDaughter[iDecay][i]->Write();
      fHistPtResidDaughter[iDecay][i]->Write();
      fHistMassResidDaughter[iDecay][i]->Write();
      fHistEResidDaughter[iDecay][i]->Write();
    }
  }
  /////////////////////////////////
  /////////////////////////////////
  TDirectory* MeanResidPtDepPlots_Dir = Resid_Dir->mkdir("MeanResidPtDepPlots");
  std::array<TDirectory*, NUM_OF_DECAYS> MotherMeanResidPtDepPlots_Dir;
  std::array<std::array<TDirectory*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> DaughtersMeanResidPtDepPlots_Dir;
  for (int iDecay = 0; iDecay < NUM_OF_DECAYS; iDecay++) {
    MotherMeanResidPtDepPlots_Dir[iDecay] = MeanResidPtDepPlots_Dir->mkdir(Form("Mother_%i", MOTHER_PDG_IDS[iDecay]));
    for (int i = 0; i < NUM_OF_DAUGHTERS[iDecay]; i++) {
      DaughtersMeanResidPtDepPlots_Dir[iDecay][i] = MeanResidPtDepPlots_Dir->mkdir(
        Form("Daughter%i_%i->%i", i + 1, MOTHER_PDG_IDS[iDecay], DAUGHTER_PDG_IDS[iDecay][i]));
    }

    MotherMeanResidPtDepPlots_Dir[iDecay]->cd();
    fGraphPtXMeanResidMother[iDecay]->Write();
    fGraphPtYMeanResidMother[iDecay]->Write();
    fGraphPtZMeanResidMother[iDecay]->Write();
    fGraphPtPxMeanResidMother[iDecay]->Write();
    fGraphPtPyMeanResidMother[iDecay]->Write();
    fGraphPtPzMeanResidMother[iDecay]->Write();
    fGraphPtPtMeanResidMother[iDecay]->Write();
    fGraphPtMassMeanResidMother[iDecay]->Write();
    fGraphPtEMeanResidMother[iDecay]->Write();

    for (int i = 0; i < NUM_OF_DAUGHTERS[iDecay]; i++) {
      DaughtersMeanResidPtDepPlots_Dir[iDecay][i]->cd();
      fGraphPtXMeanResidDaughter[iDecay][i]->Write();
      fGraphPtYMeanResidDaughter[iDecay][i]->Write();
      fGraphPtZMeanResidDaughter[iDecay][i]->Write();
      fGraphPtPxMeanResidDaughter[iDecay][i]->Write();
      fGraphPtPyMeanResidDaughter[iDecay][i]->Write();
      fGraphPtPzMeanResidDaughter[iDecay][i]->Write();
      fGraphPtPtMeanResidDaughter[iDecay][i]->Write();
      fGraphPtMassMeanResidDaughter[iDecay][i]->Write();
      fGraphPtEMeanResidDaughter[iDecay][i]->Write();
    }
  }
  /////////////////////////////////
  /////////////////////////////////
  TDirectory* ResolPtDepPlots_Dir = Resid_Dir->mkdir("ResolPtDepPlots");

  std::array<TDirectory*, NUM_OF_DECAYS> MotherWidthPtDepPlots_Dir;
  std::array<std::array<TDirectory*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> DaughtersWidthPtDepPlots_Dir;
  for (int iDecay = 0; iDecay < NUM_OF_DECAYS; iDecay++) {
    MotherWidthPtDepPlots_Dir[iDecay] = ResolPtDepPlots_Dir->mkdir(Form("Mother_%i", MOTHER_PDG_IDS[iDecay]));
    for (int i = 0; i < NUM_OF_DAUGHTERS[iDecay]; i++) {
      DaughtersWidthPtDepPlots_Dir[iDecay][i] = ResolPtDepPlots_Dir->mkdir(
        Form("Daughter%i_%i->%i", i + 1, MOTHER_PDG_IDS[iDecay], DAUGHTER_PDG_IDS[iDecay][i]));
    }

    MotherWidthPtDepPlots_Dir[iDecay]->cd();
    fGraphPtXResolMother[iDecay]->Write();
    fGraphPtYResolMother[iDecay]->Write();
    fGraphPtZResolMother[iDecay]->Write();
    fGraphPtPxResolMother[iDecay]->Write();
    fGraphPtPyResolMother[iDecay]->Write();
    fGraphPtPzResolMother[iDecay]->Write();
    fGraphPtPtResolMother[iDecay]->Write();
    fGraphPtMassResolMother[iDecay]->Write();
    fGraphPtEResolMother[iDecay]->Write();

    for (int i = 0; i < NUM_OF_DAUGHTERS[iDecay]; i++) {
      DaughtersWidthPtDepPlots_Dir[iDecay][i]->cd();
      fGraphPtXResolDaughter[iDecay][i]->Write();
      fGraphPtYResolDaughter[iDecay][i]->Write();
      fGraphPtZResolDaughter[iDecay][i]->Write();
      fGraphPtPxResolDaughter[iDecay][i]->Write();
      fGraphPtPyResolDaughter[iDecay][i]->Write();
      fGraphPtPzResolDaughter[iDecay][i]->Write();
      fGraphPtPtResolDaughter[iDecay][i]->Write();
      fGraphPtMassResolDaughter[iDecay][i]->Write();
      fGraphPtEResolDaughter[iDecay][i]->Write();
    }
  }

  TCanvas* cResids[NUM_OF_DECAYS];
  for (int iDecay = 0; iDecay < NUM_OF_DECAYS; iDecay++) {
    cResids[iDecay] =
      new TCanvas(Form("Resids_%i", MOTHER_PDG_IDS[iDecay]), Form("Resids_%i", MOTHER_PDG_IDS[iDecay]), 1600, 1000);
    cResids[iDecay]->Divide(3, 2);
    cResids[iDecay]->cd(1);
    fHistXResidMotherLst[iDecay]->Draw();
    cResids[iDecay]->cd(2);
    fHistZResidMotherLst[iDecay]->Draw();
    cResids[iDecay]->cd(3);
    fHistPxResidMotherLst[iDecay]->Draw();
    cResids[iDecay]->cd(4);
    fHistPtResidMotherLst[iDecay]->Draw();
    cResids[iDecay]->cd(5);
    fHistMassResidMotherLst[iDecay]->Draw();
    cResids[iDecay]->cd(6);
    fHistEResidMotherLst[iDecay]->Draw();
    cResids[iDecay]->SaveAs(Form("Resids_%i.C", MOTHER_PDG_IDS[iDecay]));
    cResids[iDecay]->SaveAs(Form("Resids_%i.pdf", MOTHER_PDG_IDS[iDecay]));
  }
}
