#pragma once

#include "../ConfigConstants.h"
#include "MakeGraphsFromTH2.h"
#include "OutputObjInterface.h"
#include "TCanvas.h"
#include "TH2.h"
#include <array>

class PullPtDepHistos : public OutputObjInterface {
 public:
  PullPtDepHistos(TFile& outputFile) : OutputObjInterface(outputFile) {};
  ~PullPtDepHistos() {};

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
  TH2F* fHistPtXPullMotherFst[NUM_OF_DECAYS];
  TH2F* fHistPtYPullMotherFst[NUM_OF_DECAYS];
  TH2F* fHistPtZPullMotherFst[NUM_OF_DECAYS];
  TH2F* fHistPtPxPullMotherFst[NUM_OF_DECAYS];
  TH2F* fHistPtPyPullMotherFst[NUM_OF_DECAYS];
  TH2F* fHistPtPzPullMotherFst[NUM_OF_DECAYS];
  TH2F* fHistPtPtPullMotherFst[NUM_OF_DECAYS];
  TH2F* fHistPtMassPullMotherFst[NUM_OF_DECAYS];
  TH2F* fHistPtEPullMotherFst[NUM_OF_DECAYS];

  TH2F* fHistPtXPullMotherLst[NUM_OF_DECAYS];
  TH2F* fHistPtYPullMotherLst[NUM_OF_DECAYS];
  TH2F* fHistPtZPullMotherLst[NUM_OF_DECAYS];
  TH2F* fHistPtPxPullMotherLst[NUM_OF_DECAYS];
  TH2F* fHistPtPyPullMotherLst[NUM_OF_DECAYS];
  TH2F* fHistPtPzPullMotherLst[NUM_OF_DECAYS];
  TH2F* fHistPtPtPullMotherLst[NUM_OF_DECAYS];
  TH2F* fHistPtMassPullMotherLst[NUM_OF_DECAYS];
  TH2F* fHistPtEPullMotherLst[NUM_OF_DECAYS];

  std::array<std::array<TH2F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistPtXPullDaughter;
  std::array<std::array<TH2F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistPtYPullDaughter;
  std::array<std::array<TH2F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistPtZPullDaughter;
  std::array<std::array<TH2F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistPtPxPullDaughter;
  std::array<std::array<TH2F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistPtPyPullDaughter;
  std::array<std::array<TH2F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistPtPzPullDaughter;
  std::array<std::array<TH2F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistPtPtPullDaughter;
  std::array<std::array<TH2F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistPtMassPullDaughter;
  std::array<std::array<TH2F*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistPtEPullDaughter;

  //
  TH1D* fHistXPullMotherFst[NUM_OF_DECAYS];
  TH1D* fHistYPullMotherFst[NUM_OF_DECAYS];
  TH1D* fHistZPullMotherFst[NUM_OF_DECAYS];
  TH1D* fHistPxPullMotherFst[NUM_OF_DECAYS];
  TH1D* fHistPyPullMotherFst[NUM_OF_DECAYS];
  TH1D* fHistPzPullMotherFst[NUM_OF_DECAYS];
  TH1D* fHistPtPullMotherFst[NUM_OF_DECAYS];
  TH1D* fHistMassPullMotherFst[NUM_OF_DECAYS];
  TH1D* fHistEPullMotherFst[NUM_OF_DECAYS];

  TH1D* fHistXPullMotherLst[NUM_OF_DECAYS];
  TH1D* fHistYPullMotherLst[NUM_OF_DECAYS];
  TH1D* fHistZPullMotherLst[NUM_OF_DECAYS];
  TH1D* fHistPxPullMotherLst[NUM_OF_DECAYS];
  TH1D* fHistPyPullMotherLst[NUM_OF_DECAYS];
  TH1D* fHistPzPullMotherLst[NUM_OF_DECAYS];
  TH1D* fHistPtPullMotherLst[NUM_OF_DECAYS];
  TH1D* fHistMassPullMotherLst[NUM_OF_DECAYS];
  TH1D* fHistEPullMotherLst[NUM_OF_DECAYS];

  std::array<std::array<TH1D*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistXPullDaughter;
  std::array<std::array<TH1D*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistYPullDaughter;
  std::array<std::array<TH1D*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistZPullDaughter;
  std::array<std::array<TH1D*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistPxPullDaughter;
  std::array<std::array<TH1D*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistPyPullDaughter;
  std::array<std::array<TH1D*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistPzPullDaughter;
  std::array<std::array<TH1D*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistPtPullDaughter;
  std::array<std::array<TH1D*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistMassPullDaughter;
  std::array<std::array<TH1D*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fHistEPullDaughter;

  //
  TGraphErrors* fGraphPtXMeanPullMother[NUM_OF_DECAYS];
  TGraphErrors* fGraphPtYMeanPullMother[NUM_OF_DECAYS];
  TGraphErrors* fGraphPtZMeanPullMother[NUM_OF_DECAYS];
  TGraphErrors* fGraphPtPxMeanPullMother[NUM_OF_DECAYS];
  TGraphErrors* fGraphPtPyMeanPullMother[NUM_OF_DECAYS];
  TGraphErrors* fGraphPtPzMeanPullMother[NUM_OF_DECAYS];
  TGraphErrors* fGraphPtPtMeanPullMother[NUM_OF_DECAYS];
  TGraphErrors* fGraphPtMassMeanPullMother[NUM_OF_DECAYS];
  TGraphErrors* fGraphPtEMeanPullMother[NUM_OF_DECAYS];

  std::array<std::array<TGraphErrors*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fGraphPtXMeanPullDaughter;
  std::array<std::array<TGraphErrors*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fGraphPtYMeanPullDaughter;
  std::array<std::array<TGraphErrors*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fGraphPtZMeanPullDaughter;
  std::array<std::array<TGraphErrors*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fGraphPtPxMeanPullDaughter;
  std::array<std::array<TGraphErrors*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fGraphPtPyMeanPullDaughter;
  std::array<std::array<TGraphErrors*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fGraphPtPzMeanPullDaughter;
  std::array<std::array<TGraphErrors*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fGraphPtPtMeanPullDaughter;
  std::array<std::array<TGraphErrors*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fGraphPtMassMeanPullDaughter;
  std::array<std::array<TGraphErrors*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fGraphPtEMeanPullDaughter;

  //
  TGraphErrors* fGraphPtXPullWidthMother[NUM_OF_DECAYS];
  TGraphErrors* fGraphPtYPullWidthMother[NUM_OF_DECAYS];
  TGraphErrors* fGraphPtZPullWidthMother[NUM_OF_DECAYS];
  TGraphErrors* fGraphPtPxPullWidthMother[NUM_OF_DECAYS];
  TGraphErrors* fGraphPtPyPullWidthMother[NUM_OF_DECAYS];
  TGraphErrors* fGraphPtPzPullWidthMother[NUM_OF_DECAYS];
  TGraphErrors* fGraphPtPtPullWidthMother[NUM_OF_DECAYS];
  TGraphErrors* fGraphPtMassPullWidthMother[NUM_OF_DECAYS];
  TGraphErrors* fGraphPtEPullWidthMother[NUM_OF_DECAYS];

  std::array<std::array<TGraphErrors*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fGraphPtXPullWidthDaughter;
  std::array<std::array<TGraphErrors*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fGraphPtYPullWidthDaughter;
  std::array<std::array<TGraphErrors*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fGraphPtZPullWidthDaughter;
  std::array<std::array<TGraphErrors*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fGraphPtPxPullWidthDaughter;
  std::array<std::array<TGraphErrors*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fGraphPtPyPullWidthDaughter;
  std::array<std::array<TGraphErrors*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fGraphPtPzPullWidthDaughter;
  std::array<std::array<TGraphErrors*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fGraphPtPtPullWidthDaughter;
  std::array<std::array<TGraphErrors*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fGraphPtMassPullWidthDaughter;
  std::array<std::array<TGraphErrors*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> fGraphPtEPullWidthDaughter;
};

////////////////////////////////////////////////
////---------- method implementations ----------
////////////////////////////////////////////////
void PullPtDepHistos::Create()
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
    fHistPtXPullMotherFst[iDecay] = new TH2F(Form("fHistPtXPullMotherFst_%i", MOTHER_PDG_IDS[iDecay]),
                                             "Pt-X Pull histo; p_{T}, GeV; X Pull", LogBinsX, LogWidthX, 100, -10, 10);
    fHistPtYPullMotherFst[iDecay] = new TH2F(Form("fHistPtYPullMotherFst_%i", MOTHER_PDG_IDS[iDecay]),
                                             "Pt-Y Pull histo; p_{T}, GeV; Y Pull", LogBinsX, LogWidthX, 100, -10, 10);
    fHistPtZPullMotherFst[iDecay] = new TH2F(Form("fHistPtZPullMotherFst_%i", MOTHER_PDG_IDS[iDecay]),
                                             "Pt-Z Pull histo; p_{T}, GeV; Z Pull", LogBinsX, LogWidthX, 100, -10, 10);
    fHistPtPxPullMotherFst[iDecay] =
      new TH2F(Form("fHistPtPxPullMotherFst_%i", MOTHER_PDG_IDS[iDecay]), "Pt-Px Pull histo; p_{T}, GeV; Px Pull",
               LogBinsX, LogWidthX, 50, -15, 15);
    fHistPtPyPullMotherFst[iDecay] =
      new TH2F(Form("fHistPtPyPullMotherFst_%i", MOTHER_PDG_IDS[iDecay]), "Pt-Py Pull histo; p_{T}, GeV; Py Pull",
               LogBinsX, LogWidthX, 50, -15, 15);
    fHistPtPzPullMotherFst[iDecay] =
      new TH2F(Form("fHistPtPzPullMotherFst_%i", MOTHER_PDG_IDS[iDecay]), "Pt-Pz Pull histo; p_{T}, GeV; Pz Pull",
               LogBinsX, LogWidthX, 50, -15, 15);
    fHistPtPtPullMotherFst[iDecay] =
      new TH2F(Form("fHistPtPtPullMotherFst_%i", MOTHER_PDG_IDS[iDecay]), "Pt-Pt Pull histo; p_{T}, GeV; Pt Pull",
               LogBinsX, LogWidthX, 50, -15, 15);
    fHistPtMassPullMotherFst[iDecay] =
      new TH2F(Form("fHistPtMassPullMotherFst_%i", MOTHER_PDG_IDS[iDecay]), "Pt-Mass Pull histo; p_{T}, GeV; Mass Pull",
               LogBinsX, LogWidthX, 50, -5, 5);
    fHistPtEPullMotherFst[iDecay] = new TH2F(Form("fHistPtEPullMotherFst_%i", MOTHER_PDG_IDS[iDecay]),
                                             "Pt-E Pull histo; p_{T}, GeV; E Pull", LogBinsX, LogWidthX, 50, -15, 15);
    fHistPtXPullMotherLst[iDecay] = new TH2F(Form("fHistPtXPullMotherLst_%i", MOTHER_PDG_IDS[iDecay]),
                                             "Pt-X Pull histo; p_{T}, GeV; X Pull", LogBinsX, LogWidthX, 100, -10, 10);
    fHistPtYPullMotherLst[iDecay] = new TH2F(Form("fHistPtYPullMotherLst_%i", MOTHER_PDG_IDS[iDecay]),
                                             "Pt-Y Pull histo; p_{T}, GeV; Y Pull", LogBinsX, LogWidthX, 100, -10, 10);
    fHistPtZPullMotherLst[iDecay] = new TH2F(Form("fHistPtZPullMotherLst_%i", MOTHER_PDG_IDS[iDecay]),
                                             "Pt-Z Pull histo; p_{T}, GeV; Z Pull", LogBinsX, LogWidthX, 100, -10, 10);
    fHistPtPxPullMotherLst[iDecay] =
      new TH2F(Form("fHistPtPxPullMotherLst_%i", MOTHER_PDG_IDS[iDecay]), "Pt-Px Pull histo; p_{T}, GeV; Px Pull",
               LogBinsX, LogWidthX, 50, -15, 15);
    fHistPtPyPullMotherLst[iDecay] =
      new TH2F(Form("fHistPtPyPullMotherLst_%i", MOTHER_PDG_IDS[iDecay]), "Pt-Py Pull histo; p_{T}, GeV; Py Pull",
               LogBinsX, LogWidthX, 50, -15, 15);
    fHistPtPzPullMotherLst[iDecay] =
      new TH2F(Form("fHistPtPzPullMotherLst_%i", MOTHER_PDG_IDS[iDecay]), "Pt-Pz Pull histo; p_{T}, GeV; Pz Pull",
               LogBinsX, LogWidthX, 50, -15, 15);
    fHistPtPtPullMotherLst[iDecay] =
      new TH2F(Form("fHistPtPtPullMotherLst_%i", MOTHER_PDG_IDS[iDecay]), "Pt-Pt Pull histo; p_{T}, GeV; Pt Pull",
               LogBinsX, LogWidthX, 50, -15, 15);
    fHistPtMassPullMotherLst[iDecay] =
      new TH2F(Form("fHistPtMassPullMotherLst_%i", MOTHER_PDG_IDS[iDecay]), "Pt-Mass Pull histo; p_{T}, GeV; Mass Pull",
               LogBinsX, LogWidthX, 50, -5, 5);
    fHistPtEPullMotherLst[iDecay] = new TH2F(Form("fHistPtEPullMotherLst_%i", MOTHER_PDG_IDS[iDecay]),
                                             "Pt-E Pull histo; p_{T}, GeV; E Pull", LogBinsX, LogWidthX, 50, -15, 15);

    for (int i = 0; i < NUM_OF_DAUGHTERS[iDecay]; i++) {
      fHistPtXPullDaughter[iDecay][i] =
        new TH2F(Form("fHistPtXPullDaughter%i_%i", MOTHER_PDG_IDS[iDecay], i + 1),
                 "Pt-X Pull histo; p_{T}, GeV; X Pull", LogBinsX, LogWidthX, 100, -10, 10);
      fHistPtYPullDaughter[iDecay][i] =
        new TH2F(Form("fHistPtYPullDaughter%i_%i", MOTHER_PDG_IDS[iDecay], i + 1),
                 "Pt-Y Pull histo; p_{T}, GeV; Y Pull", LogBinsX, LogWidthX, 100, -10, 10);
      fHistPtZPullDaughter[iDecay][i] =
        new TH2F(Form("fHistPtZPullDaughter%i_%i", MOTHER_PDG_IDS[iDecay], i + 1),
                 "Pt-Z Pull histo; p_{T}, GeV; Z Pull", LogBinsX, LogWidthX, 100, -10, 10);
      fHistPtPxPullDaughter[iDecay][i] =
        new TH2F(Form("fHistPtPxPullDaughter%i_%i", MOTHER_PDG_IDS[iDecay], i + 1),
                 "Pt-Px Pull histo; p_{T}, GeV; Px Pull", LogBinsX, LogWidthX, 50, -15, 15);
      fHistPtPyPullDaughter[iDecay][i] =
        new TH2F(Form("fHistPtPyPullDaughter%i_%i", MOTHER_PDG_IDS[iDecay], i + 1),
                 "Pt-Py Pull histo; p_{T}, GeV; Py Pull", LogBinsX, LogWidthX, 50, -15, 15);
      fHistPtPzPullDaughter[iDecay][i] =
        new TH2F(Form("fHistPtPzPullDaughter%i_%i", MOTHER_PDG_IDS[iDecay], i + 1),
                 "Pt-Pz Pull histo; p_{T}, GeV; Pz Pull", LogBinsX, LogWidthX, 50, -15, 15);
      fHistPtPtPullDaughter[iDecay][i] =
        new TH2F(Form("fHistPtPtPullDaughter%i_%i", MOTHER_PDG_IDS[iDecay], i + 1),
                 "Pt-Pt Pull histo; p_{T}, GeV; Pt Pull", LogBinsX, LogWidthX, 50, -15, 15);
      fHistPtMassPullDaughter[iDecay][i] =
        new TH2F(Form("fHistPtMassPullDaughter%i_%i", MOTHER_PDG_IDS[iDecay], i + 1),
                 "Pt-Mass Pull histo; p_{T}, GeV; Mass Pull", LogBinsX, LogWidthX, 50, -0.4, 0.4);
      fHistPtEPullDaughter[iDecay][i] =
        new TH2F(Form("fHistPtEPullMother%i_%i", MOTHER_PDG_IDS[iDecay], i + 1), "Pt-E Pull histo; p_{T}, GeV; E Pull",
                 LogBinsX, LogWidthX, 50, -15, 15);
    }
    //
    fHistXPullMotherFst[iDecay] =
      new TH1D(Form("fHistXPullMotherFst_%i", MOTHER_PDG_IDS[iDecay]), "X Pull histo; X Pull; counts", 100, -10, 10);
    fHistYPullMotherFst[iDecay] =
      new TH1D(Form("fHistYPullMotherFst_%i", MOTHER_PDG_IDS[iDecay]), "Y Pull histo; Y Pull; counts", 100, -10, 10);
    fHistZPullMotherFst[iDecay] =
      new TH1D(Form("fHistZPullMotherFst_%i", MOTHER_PDG_IDS[iDecay]), "Z Pull histo; Z Pull; counts", 100, -10, 10);
    fHistPxPullMotherFst[iDecay] =
      new TH1D(Form("fHistPxPullMotherFst_%i", MOTHER_PDG_IDS[iDecay]), "Px Pull histo; Px Pull; counts", 50, -15, 15);
    fHistPyPullMotherFst[iDecay] =
      new TH1D(Form("fHistPyPullMotherFst_%i", MOTHER_PDG_IDS[iDecay]), "Py Pull histo; Py Pull; counts", 50, -15, 15);
    fHistPzPullMotherFst[iDecay] =
      new TH1D(Form("fHistPzPullMotherFst_%i", MOTHER_PDG_IDS[iDecay]), "Pz Pull histo; Pz Pull; counts", 50, -15, 15);
    fHistPtPullMotherFst[iDecay]   = new TH1D(Form("fHistPtPullMotherFst_%i", MOTHER_PDG_IDS[iDecay]),
                                              "Pt - Pull histo; Pt Pull; counts", 50, -15, 15);
    fHistMassPullMotherFst[iDecay] = new TH1D(Form("fHistMassPullMotherFst_%i", MOTHER_PDG_IDS[iDecay]),
                                              "Mass Pull histo; Mass Pull; counts", 50, -15, 15);
    fHistEPullMotherFst[iDecay] =
      new TH1D(Form("fHistEPullMotherFst_%i", MOTHER_PDG_IDS[iDecay]), "E Pull histo; E Pull; counts", 50, -15, 15);

    fHistXPullMotherLst[iDecay] =
      new TH1D(Form("fHistXPullMotherLst_%i", MOTHER_PDG_IDS[iDecay]), "X Pull histo; X Pull; counts", 100, -10, 10);
    fHistYPullMotherLst[iDecay] =
      new TH1D(Form("fHistYPullMotherLst_%i", MOTHER_PDG_IDS[iDecay]), "Y Pull histo; Y Pull; counts", 100, -10, 10);
    fHistZPullMotherLst[iDecay] =
      new TH1D(Form("fHistZPullMotherLst_%i", MOTHER_PDG_IDS[iDecay]), "Z Pull histo; Z Pull; counts", 100, -10, 10);
    fHistPxPullMotherLst[iDecay] =
      new TH1D(Form("fHistPxPullMotherLst_%i", MOTHER_PDG_IDS[iDecay]), "Px Pull histo; Px Pull; counts", 50, -15, 15);
    fHistPyPullMotherLst[iDecay] =
      new TH1D(Form("fHistPyPullMotherLst_%i", MOTHER_PDG_IDS[iDecay]), "Py Pull histo; Py Pull; counts", 50, -15, 15);
    fHistPzPullMotherLst[iDecay] =
      new TH1D(Form("fHistPzPullMotherLst_%i", MOTHER_PDG_IDS[iDecay]), "Pz Pull histo; Pz Pull; counts", 50, -15, 15);
    fHistPtPullMotherLst[iDecay]   = new TH1D(Form("fHistPtPullMotherLst_%i", MOTHER_PDG_IDS[iDecay]),
                                              "Pt - Pull histo; Pt Pull; counts", 50, -15, 15);
    fHistMassPullMotherLst[iDecay] = new TH1D(Form("fHistMassPullMotherLst_%i", MOTHER_PDG_IDS[iDecay]),
                                              "Mass Pull histo; Mass Pull; counts", 50, -15, 15);
    fHistEPullMotherLst[iDecay] =
      new TH1D(Form("fHistEPullMotherLst_%i", MOTHER_PDG_IDS[iDecay]), "E Pull histo; E Pull; counts", 50, -15, 15);

    for (int i = 0; i < NUM_OF_DAUGHTERS[iDecay]; i++) {
      fHistXPullDaughter[iDecay][i]    = new TH1D(Form("fHistXPullDaughter%i_%i", MOTHER_PDG_IDS[iDecay], i),
                                                  "X Pull histo; X Pull; counts", 100, -10, 10);
      fHistYPullDaughter[iDecay][i]    = new TH1D(Form("fHistYPullDaughter%i_%i", MOTHER_PDG_IDS[iDecay], i),
                                                  "Y Pull histo; Y Pull; counts", 100, -10, 10);
      fHistZPullDaughter[iDecay][i]    = new TH1D(Form("fHistZPullDaughter%i_%i", MOTHER_PDG_IDS[iDecay], i),
                                                  "Z Pull histo; Z Pull; counts", 100, -10, 10);
      fHistPxPullDaughter[iDecay][i]   = new TH1D(Form("fHistPxPullDaughter%i_%i", MOTHER_PDG_IDS[iDecay], i),
                                                  "Px Pull histo; Px Pull; counts", 50, -15, 15);
      fHistPyPullDaughter[iDecay][i]   = new TH1D(Form("fHistPyPullDaughter%i_%i", MOTHER_PDG_IDS[iDecay], i),
                                                  "Py Pull histo; Py Pull; counts", 50, -15, 15);
      fHistPzPullDaughter[iDecay][i]   = new TH1D(Form("fHistPzPullDaughter%i_%i", MOTHER_PDG_IDS[iDecay], i),
                                                  "Pz Pull histo; Pz Pull; counts", 50, -15, 15);
      fHistPtPullDaughter[iDecay][i]   = new TH1D(Form("fHistPtPullDaughter%i_%i", MOTHER_PDG_IDS[iDecay], i),
                                                  "Pt Pull histo;  Pull; counts", 50, -15, 15);
      fHistMassPullDaughter[iDecay][i] = new TH1D(Form("fHistMassPullDaughter%i_%i", MOTHER_PDG_IDS[iDecay], i),
                                                  "Mass Pull histo; Mass Pull; counts", 50, -15, 15);
      fHistEPullDaughter[iDecay][i]    = new TH1D(Form("fHistEPullDaughter%i_%i", MOTHER_PDG_IDS[iDecay], i),
                                                  "E Pull histo; E Pull; counts", 50, -15, 15);
    }
    //
    fGraphPtXMeanPullMother[iDecay]    = new TGraphErrors();
    fGraphPtYMeanPullMother[iDecay]    = new TGraphErrors();
    fGraphPtZMeanPullMother[iDecay]    = new TGraphErrors();
    fGraphPtPxMeanPullMother[iDecay]   = new TGraphErrors();
    fGraphPtPyMeanPullMother[iDecay]   = new TGraphErrors();
    fGraphPtPzMeanPullMother[iDecay]   = new TGraphErrors();
    fGraphPtPtMeanPullMother[iDecay]   = new TGraphErrors();
    fGraphPtMassMeanPullMother[iDecay] = new TGraphErrors();
    fGraphPtEMeanPullMother[iDecay]    = new TGraphErrors();

    for (int i = 0; i < NUM_OF_DAUGHTERS[iDecay]; i++) {
      fGraphPtXMeanPullDaughter[iDecay][i]    = new TGraphErrors();
      fGraphPtYMeanPullDaughter[iDecay][i]    = new TGraphErrors();
      fGraphPtZMeanPullDaughter[iDecay][i]    = new TGraphErrors();
      fGraphPtPxMeanPullDaughter[iDecay][i]   = new TGraphErrors();
      fGraphPtPyMeanPullDaughter[iDecay][i]   = new TGraphErrors();
      fGraphPtPzMeanPullDaughter[iDecay][i]   = new TGraphErrors();
      fGraphPtPtMeanPullDaughter[iDecay][i]   = new TGraphErrors();
      fGraphPtMassMeanPullDaughter[iDecay][i] = new TGraphErrors();
      fGraphPtEMeanPullDaughter[iDecay][i]    = new TGraphErrors();
    }
    //
    fGraphPtXPullWidthMother[iDecay]    = new TGraphErrors();
    fGraphPtYPullWidthMother[iDecay]    = new TGraphErrors();
    fGraphPtZPullWidthMother[iDecay]    = new TGraphErrors();
    fGraphPtPxPullWidthMother[iDecay]   = new TGraphErrors();
    fGraphPtPyPullWidthMother[iDecay]   = new TGraphErrors();
    fGraphPtPzPullWidthMother[iDecay]   = new TGraphErrors();
    fGraphPtPtPullWidthMother[iDecay]   = new TGraphErrors();
    fGraphPtMassPullWidthMother[iDecay] = new TGraphErrors();
    fGraphPtEPullWidthMother[iDecay]    = new TGraphErrors();
    for (int i = 0; i < NUM_OF_DAUGHTERS[iDecay]; i++) {
      fGraphPtXPullWidthDaughter[iDecay][i]    = new TGraphErrors();
      fGraphPtYPullWidthDaughter[iDecay][i]    = new TGraphErrors();
      fGraphPtZPullWidthDaughter[iDecay][i]    = new TGraphErrors();
      fGraphPtPxPullWidthDaughter[iDecay][i]   = new TGraphErrors();
      fGraphPtPyPullWidthDaughter[iDecay][i]   = new TGraphErrors();
      fGraphPtPzPullWidthDaughter[iDecay][i]   = new TGraphErrors();
      fGraphPtPtPullWidthDaughter[iDecay][i]   = new TGraphErrors();
      fGraphPtMassPullWidthDaughter[iDecay][i] = new TGraphErrors();
      fGraphPtEPullWidthDaughter[iDecay][i]    = new TGraphErrors();
    }
  }
}

void PullPtDepHistos::Fill(int iDecay, const MCParticleStruct& mcPart)
{
  //....
}
void PullPtDepHistos::Fill(int iDecay, const KFParticleStruct& kfPart)
{
  //....
}

// void PullPtDepHistos::Fill(const MCParticleStruct& mcPart, const
// KFParticleStruct& kfPartDecay, const KFParticleStruct& kfPartProduction){};

void PullPtDepHistos::Fill(int iDecay, const MCParticleStruct& mcPart, const KFParticleStruct& kfPart) {};

void PullPtDepHistos::Fill(int iDecay, const MCParticleStruct& mcPart, const KFParticleStruct& kfPartDecay,
                           const KFParticleStruct& kfPartProduction)
{
  if (kfPartDecay.isMother) {
    float x_resid_    = mcPart.finalX - kfPartDecay.x;
    float y_resid_    = mcPart.finalY - kfPartDecay.y;
    float z_resid_    = mcPart.finalZ - kfPartDecay.z;
    float px_resid_   = mcPart.finalPX - kfPartDecay.pX;
    float py_resid_   = mcPart.finalPY - kfPartDecay.pY;
    float pz_resid_   = mcPart.finalPZ - kfPartDecay.pZ;
    float pt_resid_   = mcPart.finalPT() - kfPartDecay.pT;
    float mass_resid_ = mcPart.mass - kfPartDecay.mass;
    float e_resid_    = mcPart.finalE() - kfPartDecay.e;

    float x_pull_    = x_resid_ / kfPartDecay.xErr;
    float y_pull_    = y_resid_ / kfPartDecay.yErr;
    float z_pull_    = z_resid_ / kfPartDecay.zErr;
    float px_pull_   = px_resid_ / kfPartDecay.pXErr;
    float py_pull_   = py_resid_ / kfPartDecay.pYErr;
    float pz_pull_   = pz_resid_ / kfPartDecay.pZErr;
    float pt_pull_   = pt_resid_ / kfPartDecay.pTErr;
    float mass_pull_ = mass_resid_ / kfPartDecay.massErr;
    float e_pull_    = e_resid_ / kfPartDecay.eErr;

    fHistPtXPullMotherLst[iDecay]->Fill(mcPart.finalPT(), x_pull_);
    fHistPtYPullMotherLst[iDecay]->Fill(mcPart.finalPT(), y_pull_);
    fHistPtZPullMotherLst[iDecay]->Fill(mcPart.finalPT(), z_pull_);
    fHistPtPxPullMotherLst[iDecay]->Fill(mcPart.finalPT(), px_pull_);
    fHistPtPyPullMotherLst[iDecay]->Fill(mcPart.finalPT(), py_pull_);
    fHistPtPzPullMotherLst[iDecay]->Fill(mcPart.finalPT(), pz_pull_);
    fHistPtPtPullMotherLst[iDecay]->Fill(mcPart.finalPT(), pt_pull_);
    fHistPtMassPullMotherLst[iDecay]->Fill(mcPart.finalPT(), mass_pull_);
    fHistPtEPullMotherLst[iDecay]->Fill(mcPart.finalPT(), e_pull_);

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

    float x_pull_prod_    = x_resid_prod_ / kfPartProduction.xErr;
    float y_pull_prod_    = y_resid_prod_ / kfPartProduction.yErr;
    float z_pull_prod_    = z_resid_prod_ / kfPartProduction.zErr;
    float px_pull_prod_   = px_resid_prod_ / kfPartProduction.pXErr;
    float py_pull_prod_   = py_resid_prod_ / kfPartProduction.pYErr;
    float pz_pull_prod_   = pz_resid_prod_ / kfPartProduction.pZErr;
    float pt_pull_prod_   = pt_resid_prod_ / kfPartProduction.pTErr;
    float mass_pull_prod_ = mass_resid_prod_ / kfPartProduction.massErr;
    float e_pull_prod_    = e_resid_prod_ / kfPartProduction.eErr;

    fHistPtXPullMotherFst[iDecay]->Fill(mcPart.initialPT(), x_pull_prod_);
    fHistPtYPullMotherFst[iDecay]->Fill(mcPart.initialPT(), y_pull_prod_);
    fHistPtZPullMotherFst[iDecay]->Fill(mcPart.initialPT(), z_pull_prod_);
    fHistPtPxPullMotherFst[iDecay]->Fill(mcPart.initialPT(), px_pull_prod_);
    fHistPtPyPullMotherFst[iDecay]->Fill(mcPart.initialPT(), py_pull_prod_);
    fHistPtPzPullMotherFst[iDecay]->Fill(mcPart.initialPT(), pz_pull_prod_);
    fHistPtPtPullMotherFst[iDecay]->Fill(mcPart.initialPT(), pt_pull_prod_);
    fHistPtMassPullMotherFst[iDecay]->Fill(mcPart.initialPT(), mass_pull_prod_);
    fHistPtEPullMotherFst[iDecay]->Fill(mcPart.initialPT(), e_pull_prod_);
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

    float x_pull_    = x_resid_ / kfPartDecay.xErr;
    float y_pull_    = y_resid_ / kfPartDecay.yErr;
    float z_pull_    = z_resid_ / kfPartDecay.zErr;
    float px_pull_   = px_resid_ / kfPartDecay.pXErr;
    float py_pull_   = py_resid_ / kfPartDecay.pYErr;
    float pz_pull_   = pz_resid_ / kfPartDecay.pZErr;
    float pt_pull_   = pt_resid_ / kfPartDecay.pTErr;
    float mass_pull_ = mass_resid_ / kfPartDecay.massErr;
    float e_pull_    = e_resid_ / kfPartDecay.eErr;

    fHistPtXPullDaughter[iDecay][daughterNum]->Fill(mcPart.initialPT(), x_pull_);
    fHistPtYPullDaughter[iDecay][daughterNum]->Fill(mcPart.initialPT(), y_pull_);
    fHistPtZPullDaughter[iDecay][daughterNum]->Fill(mcPart.initialPT(), z_pull_);
    fHistPtPxPullDaughter[iDecay][daughterNum]->Fill(mcPart.initialPT(), px_pull_);
    fHistPtPyPullDaughter[iDecay][daughterNum]->Fill(mcPart.initialPT(), py_pull_);
    fHistPtPzPullDaughter[iDecay][daughterNum]->Fill(mcPart.initialPT(), pz_pull_);
    fHistPtPtPullDaughter[iDecay][daughterNum]->Fill(mcPart.initialPT(), pt_pull_);
    fHistPtMassPullDaughter[iDecay][daughterNum]->Fill(mcPart.initialPT(), mass_pull_);
    fHistPtEPullDaughter[iDecay][daughterNum]->Fill(mcPart.initialPT(), e_pull_);
  }
}

void PullPtDepHistos::PreWriteProcess()
{
  for (int iDecay = 0; iDecay < NUM_OF_DECAYS; iDecay++) {
    fHistXPullMotherFst[iDecay]    = fHistPtXPullMotherFst[iDecay]->ProjectionY();
    fHistYPullMotherFst[iDecay]    = fHistPtYPullMotherFst[iDecay]->ProjectionY();
    fHistZPullMotherFst[iDecay]    = fHistPtZPullMotherFst[iDecay]->ProjectionY();
    fHistPxPullMotherFst[iDecay]   = fHistPtPxPullMotherFst[iDecay]->ProjectionY();
    fHistPyPullMotherFst[iDecay]   = fHistPtPyPullMotherFst[iDecay]->ProjectionY();
    fHistPzPullMotherFst[iDecay]   = fHistPtPzPullMotherFst[iDecay]->ProjectionY();
    fHistPtPullMotherFst[iDecay]   = fHistPtPtPullMotherFst[iDecay]->ProjectionY();
    fHistMassPullMotherFst[iDecay] = fHistPtMassPullMotherFst[iDecay]->ProjectionY();
    fHistEPullMotherFst[iDecay]    = fHistPtEPullMotherFst[iDecay]->ProjectionY();

    fHistXPullMotherLst[iDecay]    = fHistPtXPullMotherLst[iDecay]->ProjectionY();
    fHistYPullMotherLst[iDecay]    = fHistPtYPullMotherLst[iDecay]->ProjectionY();
    fHistZPullMotherLst[iDecay]    = fHistPtZPullMotherLst[iDecay]->ProjectionY();
    fHistPxPullMotherLst[iDecay]   = fHistPtPxPullMotherLst[iDecay]->ProjectionY();
    fHistPyPullMotherLst[iDecay]   = fHistPtPyPullMotherLst[iDecay]->ProjectionY();
    fHistPzPullMotherLst[iDecay]   = fHistPtPzPullMotherLst[iDecay]->ProjectionY();
    fHistPtPullMotherLst[iDecay]   = fHistPtPtPullMotherLst[iDecay]->ProjectionY();
    fHistMassPullMotherLst[iDecay] = fHistPtMassPullMotherLst[iDecay]->ProjectionY();
    fHistEPullMotherLst[iDecay]    = fHistPtEPullMotherLst[iDecay]->ProjectionY();

    for (int i = 0; i < NUM_OF_DAUGHTERS[iDecay]; i++) {
      fHistXPullDaughter[iDecay][i]    = fHistPtXPullDaughter[iDecay][i]->ProjectionY();
      fHistYPullDaughter[iDecay][i]    = fHistPtYPullDaughter[iDecay][i]->ProjectionY();
      fHistZPullDaughter[iDecay][i]    = fHistPtZPullDaughter[iDecay][i]->ProjectionY();
      fHistPxPullDaughter[iDecay][i]   = fHistPtPxPullDaughter[iDecay][i]->ProjectionY();
      fHistPyPullDaughter[iDecay][i]   = fHistPtPyPullDaughter[iDecay][i]->ProjectionY();
      fHistPzPullDaughter[iDecay][i]   = fHistPtPzPullDaughter[iDecay][i]->ProjectionY();
      fHistPtPullDaughter[iDecay][i]   = fHistPtPtPullDaughter[iDecay][i]->ProjectionY();
      fHistMassPullDaughter[iDecay][i] = fHistPtMassPullDaughter[iDecay][i]->ProjectionY();
      fHistEPullDaughter[iDecay][i]    = fHistPtEPullDaughter[iDecay][i]->ProjectionY();
    }
    ///////
    fGraphPtXMeanPullMother[iDecay] = MakeMeanGraphFromTH2(fHistPtXPullMotherLst[iDecay]);
    fGraphPtXMeanPullMother[iDecay]->SetNameTitle(Form("X Mean Pull_%i", MOTHER_PDG_IDS[iDecay]),
                                                  "XMeanPull; p_{T}, GeV; X Pull");
    fGraphPtYMeanPullMother[iDecay] = MakeMeanGraphFromTH2(fHistPtYPullMotherLst[iDecay]);
    fGraphPtYMeanPullMother[iDecay]->SetNameTitle(Form("Y Mean Pull_%i", MOTHER_PDG_IDS[iDecay]),
                                                  "YMeanPull; p_{T}, GeV; Y Pull");
    fGraphPtZMeanPullMother[iDecay] = MakeMeanGraphFromTH2(fHistPtZPullMotherLst[iDecay]);
    fGraphPtZMeanPullMother[iDecay]->SetNameTitle(Form("Z Mean Pull_%i", MOTHER_PDG_IDS[iDecay]),
                                                  "ZMeanPull; p_{T}, GeV; Z Pull");
    fGraphPtPxMeanPullMother[iDecay] = MakeMeanGraphFromTH2(fHistPtPxPullMotherLst[iDecay]);
    fGraphPtPxMeanPullMother[iDecay]->SetNameTitle(Form("Px Mean Pull_%i", MOTHER_PDG_IDS[iDecay]),
                                                   "PxMeanPull; p_{T}, GeV; Px Pull");
    fGraphPtPyMeanPullMother[iDecay] = MakeMeanGraphFromTH2(fHistPtPyPullMotherLst[iDecay]);
    fGraphPtPyMeanPullMother[iDecay]->SetNameTitle(Form("Py Mean Pull_%i", MOTHER_PDG_IDS[iDecay]),
                                                   "PyMeanPull; p_{T}, GeV; Py Pull");
    fGraphPtPzMeanPullMother[iDecay] = MakeMeanGraphFromTH2(fHistPtPzPullMotherLst[iDecay]);
    fGraphPtPzMeanPullMother[iDecay]->SetNameTitle(Form("Pz Mean Pull_%i", MOTHER_PDG_IDS[iDecay]),
                                                   "PzMeanPull; p_{T}, GeV; Pz Pull");
    fGraphPtPtMeanPullMother[iDecay] = MakeMeanGraphFromTH2(fHistPtPtPullMotherLst[iDecay]);
    fGraphPtPtMeanPullMother[iDecay]->SetNameTitle(Form("Pt Mean Pull_%i", MOTHER_PDG_IDS[iDecay]),
                                                   "PtMeanPull; p_{T}, GeV; Pt Pull");
    fGraphPtMassMeanPullMother[iDecay] = MakeMeanGraphFromTH2(fHistPtMassPullMotherLst[iDecay]);
    fGraphPtMassMeanPullMother[iDecay]->SetNameTitle(Form("Mass Mean Pull_%i", MOTHER_PDG_IDS[iDecay]),
                                                     "MassMeanPull; p_{T}, GeV; Mass Pull");
    fGraphPtEMeanPullMother[iDecay] = MakeMeanGraphFromTH2(fHistPtEPullMotherLst[iDecay]);
    fGraphPtEMeanPullMother[iDecay]->SetNameTitle(Form("E Mean Pull_%i", MOTHER_PDG_IDS[iDecay]),
                                                  "EMeanPull; p_{T}, GeV; E Pull");

    for (int i = 0; i < NUM_OF_DAUGHTERS[iDecay]; i++) {
      fGraphPtXMeanPullDaughter[iDecay][i] = MakeMeanGraphFromTH2(fHistPtXPullDaughter[iDecay][i]);
      fGraphPtXMeanPullDaughter[iDecay][i]->SetNameTitle(Form("X Mean Pull_%i", MOTHER_PDG_IDS[iDecay]),
                                                         "XMeanPull; p_{T}, GeV; X Pull");
      fGraphPtYMeanPullDaughter[iDecay][i] = MakeMeanGraphFromTH2(fHistPtYPullDaughter[iDecay][i]);
      fGraphPtYMeanPullDaughter[iDecay][i]->SetNameTitle(Form("Y Mean Pull_%i", MOTHER_PDG_IDS[iDecay]),
                                                         "YMeanPull; p_{T}, GeV; Y Pull");
      fGraphPtZMeanPullDaughter[iDecay][i] = MakeMeanGraphFromTH2(fHistPtZPullDaughter[iDecay][i]);
      fGraphPtZMeanPullDaughter[iDecay][i]->SetNameTitle(Form("Z Mean Pull_%i", MOTHER_PDG_IDS[iDecay]),
                                                         "ZMeanPull; p_{T}, GeV; Z Pull");
      fGraphPtPxMeanPullDaughter[iDecay][i] = MakeMeanGraphFromTH2(fHistPtPxPullDaughter[iDecay][i]);
      fGraphPtPxMeanPullDaughter[iDecay][i]->SetNameTitle(Form("Px Mean Pull_%i", MOTHER_PDG_IDS[iDecay]),
                                                          "PxMeanPull; p_{T}, GeV; Px Pull");
      fGraphPtPyMeanPullDaughter[iDecay][i] = MakeMeanGraphFromTH2(fHistPtPyPullDaughter[iDecay][i]);
      fGraphPtPyMeanPullDaughter[iDecay][i]->SetNameTitle(Form("Py Mean Pull_%i", MOTHER_PDG_IDS[iDecay]),
                                                          "PyMeanPull; p_{T}, GeV; Py Pull");
      fGraphPtPzMeanPullDaughter[iDecay][i] = MakeMeanGraphFromTH2(fHistPtPzPullDaughter[iDecay][i]);
      fGraphPtPzMeanPullDaughter[iDecay][i]->SetNameTitle(Form("Pz Mean Pull_%i", MOTHER_PDG_IDS[iDecay]),
                                                          "PzMeanPull; p_{T}, GeV; Pz Pull");
      fGraphPtPtMeanPullDaughter[iDecay][i] = MakeMeanGraphFromTH2(fHistPtPtPullDaughter[iDecay][i]);
      fGraphPtPtMeanPullDaughter[iDecay][i]->SetNameTitle(Form("Pt Mean Pull_%i", MOTHER_PDG_IDS[iDecay]),
                                                          "PtMeanPull; p_{T}, GeV; Pt Pull");
      fGraphPtMassMeanPullDaughter[iDecay][i] = MakeMeanGraphFromTH2(fHistPtMassPullDaughter[iDecay][i]);
      fGraphPtMassMeanPullDaughter[iDecay][i]->SetNameTitle(Form("Mass Mean Pull_%i", MOTHER_PDG_IDS[iDecay]),
                                                            "MassMeanPull; p_{T}, GeV; Mass Pull");
      fGraphPtEMeanPullDaughter[iDecay][i] = MakeMeanGraphFromTH2(fHistPtEPullDaughter[iDecay][i]);
      fGraphPtEMeanPullDaughter[iDecay][i]->SetNameTitle(Form("E Mean Pull_%i", MOTHER_PDG_IDS[iDecay]),
                                                         "EMeanPull; p_{T}, GeV; E Pull");
    }
    ///////
    fGraphPtXPullWidthMother[iDecay] = MakeSigmaGraphFromTH2(fHistPtXPullMotherLst[iDecay]);
    fGraphPtXPullWidthMother[iDecay]->SetNameTitle(Form("X Pull Width_%i", MOTHER_PDG_IDS[iDecay]),
                                                   "XPullWidth; p_{T}, GeV; X PullWidth");
    fGraphPtYPullWidthMother[iDecay] = MakeSigmaGraphFromTH2(fHistPtYPullMotherLst[iDecay]);
    fGraphPtYPullWidthMother[iDecay]->SetNameTitle(Form("Y Pull Width_%i", MOTHER_PDG_IDS[iDecay]),
                                                   "YPullWidth; p_{T}, GeV; Y PullWidth");
    fGraphPtZPullWidthMother[iDecay] = MakeSigmaGraphFromTH2(fHistPtZPullMotherLst[iDecay]);
    fGraphPtZPullWidthMother[iDecay]->SetNameTitle(Form("Z Pull Width_%i", MOTHER_PDG_IDS[iDecay]),
                                                   "ZPullWidth; p_{T}, GeV; Z PullWidth");
    fGraphPtPxPullWidthMother[iDecay] = MakeSigmaGraphFromTH2(fHistPtPxPullMotherLst[iDecay]);
    fGraphPtPxPullWidthMother[iDecay]->SetNameTitle(Form("Px Pull Width_%i", MOTHER_PDG_IDS[iDecay]),
                                                    "PxPullWidth; p_{T}, GeV; Px PullWidth");
    fGraphPtPyPullWidthMother[iDecay] = MakeSigmaGraphFromTH2(fHistPtPyPullMotherLst[iDecay]);
    fGraphPtPyPullWidthMother[iDecay]->SetNameTitle(Form("Py Pull Width_%i", MOTHER_PDG_IDS[iDecay]),
                                                    "PyPullWidth; p_{T}, GeV; Py PullWidth");
    fGraphPtPzPullWidthMother[iDecay] = MakeSigmaGraphFromTH2(fHistPtPzPullMotherLst[iDecay]);
    fGraphPtPzPullWidthMother[iDecay]->SetNameTitle(Form("Pz Pull Width_%i", MOTHER_PDG_IDS[iDecay]),
                                                    "PzPullWidth; p_{T}, GeV; Pz PullWidth");
    fGraphPtPtPullWidthMother[iDecay] = MakeSigmaGraphFromTH2(fHistPtPtPullMotherLst[iDecay]);
    fGraphPtPtPullWidthMother[iDecay]->SetNameTitle(Form("Pt Pull Width_%i", MOTHER_PDG_IDS[iDecay]),
                                                    "PtPullWidth; p_{T}, GeV; Pt PullWidth");
    fGraphPtMassPullWidthMother[iDecay] = MakeSigmaGraphFromTH2(fHistPtMassPullMotherLst[iDecay]);
    fGraphPtMassPullWidthMother[iDecay]->SetNameTitle(Form("Mass Pull Width_%i", MOTHER_PDG_IDS[iDecay]),
                                                      "MassPullWidth; p_{T}, GeV; Mass PullWidth");
    fGraphPtEPullWidthMother[iDecay] = MakeSigmaGraphFromTH2(fHistPtEPullMotherLst[iDecay]);
    fGraphPtEPullWidthMother[iDecay]->SetNameTitle(Form("E Pull Width_%i", MOTHER_PDG_IDS[iDecay]),
                                                   "EPullWidth; p_{T}, GeV; E PullWidth");

    for (int i = 0; i < NUM_OF_DAUGHTERS[iDecay]; i++) {
      fGraphPtXPullWidthDaughter[iDecay][i] = MakeSigmaGraphFromTH2(fHistPtXPullDaughter[iDecay][i]);
      fGraphPtXPullWidthDaughter[iDecay][i]->SetNameTitle(Form("X PullWidth_%i", MOTHER_PDG_IDS[iDecay]),
                                                          "XPullWidth; p_{T}, GeV; X PullWidth");
      fGraphPtYPullWidthDaughter[iDecay][i] = MakeSigmaGraphFromTH2(fHistPtYPullDaughter[iDecay][i]);
      fGraphPtYPullWidthDaughter[iDecay][i]->SetNameTitle(Form("Y PullWidth_%i", MOTHER_PDG_IDS[iDecay]),
                                                          "YPullWidth; p_{T}, GeV; Y PullWidth");
      fGraphPtZPullWidthDaughter[iDecay][i] = MakeSigmaGraphFromTH2(fHistPtZPullDaughter[iDecay][i]);
      fGraphPtZPullWidthDaughter[iDecay][i]->SetNameTitle(Form("Z PullWidth_%i", MOTHER_PDG_IDS[iDecay]),
                                                          "ZPullWidth; p_{T}, GeV; Z PullWidth");
      fGraphPtPxPullWidthDaughter[iDecay][i] = MakeSigmaGraphFromTH2(fHistPtPxPullDaughter[iDecay][i]);
      fGraphPtPxPullWidthDaughter[iDecay][i]->SetNameTitle(Form("Px PullWidth_%i", MOTHER_PDG_IDS[iDecay]),
                                                           "PxPullWidth; p_{T}, GeV; Px PullWidth");
      fGraphPtPyPullWidthDaughter[iDecay][i] = MakeSigmaGraphFromTH2(fHistPtPyPullDaughter[iDecay][i]);
      fGraphPtPyPullWidthDaughter[iDecay][i]->SetNameTitle(Form("Py PullWidth_%i", MOTHER_PDG_IDS[iDecay]),
                                                           "PyPullWidth; p_{T}, GeV; Py PullWidth");
      fGraphPtPzPullWidthDaughter[iDecay][i] = MakeSigmaGraphFromTH2(fHistPtPzPullDaughter[iDecay][i]);
      fGraphPtPzPullWidthDaughter[iDecay][i]->SetNameTitle(Form("Pz PullWidth_%i", MOTHER_PDG_IDS[iDecay]),
                                                           "PzPullWidth; p_{T}, GeV; Pz PullWidth");
      fGraphPtPtPullWidthDaughter[iDecay][i] = MakeSigmaGraphFromTH2(fHistPtPtPullDaughter[iDecay][i]);
      fGraphPtPtPullWidthDaughter[iDecay][i]->SetNameTitle(Form("Pt PullWidth_%i", MOTHER_PDG_IDS[iDecay]),
                                                           "PtPullWidth; p_{T}, GeV; Pt PullWidth");
      fGraphPtMassPullWidthDaughter[iDecay][i] = MakeSigmaGraphFromTH2(fHistPtMassPullDaughter[iDecay][i]);
      fGraphPtMassPullWidthDaughter[iDecay][i]->SetNameTitle(Form("Mass PullWidth_%i", MOTHER_PDG_IDS[iDecay]),
                                                             "MassPullWidth; p_{T}, GeV; Mass PullWidth");
      fGraphPtEPullWidthDaughter[iDecay][i] = MakeSigmaGraphFromTH2(fHistPtEPullDaughter[iDecay][i]);
      fGraphPtEPullWidthDaughter[iDecay][i]->SetNameTitle(Form("E PullWidth_%i", MOTHER_PDG_IDS[iDecay]),
                                                          "EPullWidth; p_{T}, GeV; E PullWidth");
    }
  }
}

bool PullPtDepHistos::Check()
{
  cout << "Checking pools..." << endl;
  bool pullsok  = true;
  string errMsg = "";
  for (int iDecay = 0; iDecay < NUM_OF_DECAYS; iDecay++) {
    // check mean and width of pulls (x,y,z,px,py,pz,pt,m,e
    double xMean    = fHistXPullMotherLst[iDecay]->GetMean();
    double yMean    = fHistYPullMotherLst[iDecay]->GetMean();
    double zMean    = fHistZPullMotherLst[iDecay]->GetMean();
    double pxMean   = fHistPxPullMotherLst[iDecay]->GetMean();
    double pyMean   = fHistPyPullMotherLst[iDecay]->GetMean();
    double pzMean   = fHistPzPullMotherLst[iDecay]->GetMean();
    double ptMean   = fHistPtPullMotherLst[iDecay]->GetMean();
    double mMean    = fHistMassPullMotherLst[iDecay]->GetMean();
    double eMean    = fHistEPullMotherLst[iDecay]->GetMean();
    double xStdDev  = fHistXPullMotherLst[iDecay]->GetStdDev();
    double yStdDev  = fHistYPullMotherLst[iDecay]->GetStdDev();
    double zStdDev  = fHistZPullMotherLst[iDecay]->GetStdDev();
    double pxStdDev = fHistPxPullMotherLst[iDecay]->GetStdDev();
    double pyStdDev = fHistPyPullMotherLst[iDecay]->GetStdDev();
    double pzStdDev = fHistPzPullMotherLst[iDecay]->GetStdDev();
    double ptStdDev = fHistPtPullMotherLst[iDecay]->GetStdDev();
    double mStdDev  = fHistMassPullMotherLst[iDecay]->GetStdDev();
    double eStdDev  = fHistEPullMotherLst[iDecay]->GetStdDev();

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
      pullsok = false;
      errMsg += Form("\nERROR: Decay %i: Pull x is biased: xMean = %F", MOTHER_PDG_IDS[iDecay], xMean);
    }
    if (fabs(yMean) > fabs(err_yMean)) {
      pullsok = false;
      errMsg += Form("\nERROR: Decay %i: Pull y is biased: yMean = %F", MOTHER_PDG_IDS[iDecay], yMean);
    }
    if (fabs(zMean) > fabs(err_zMean)) {
      pullsok = false;
      errMsg += Form("\nERROR: Decay %i: Pull z is biased: zMean = %F", MOTHER_PDG_IDS[iDecay], zMean);
    }
    if (fabs(pxMean) > fabs(err_pxMean)) {
      pullsok = false;
      errMsg += Form("\nERROR: Decay %i: Pull px is biased: pxMean = %F", MOTHER_PDG_IDS[iDecay], pxMean);
    }
    if (fabs(pyMean) > fabs(err_pyMean)) {
      pullsok = false;
      errMsg += Form("\nERROR: Decay %i: Pull py is biased: pyMean = %F", MOTHER_PDG_IDS[iDecay], pyMean);
    }
    if (fabs(pzMean) > fabs(err_pzMean)) {
      pullsok = false;
      errMsg += Form("\nERROR: Decay %i: Pull pz is biased: pzMean = %F", MOTHER_PDG_IDS[iDecay], pzMean);
    }
    if (fabs(ptMean) > fabs(err_ptMean)) {
      pullsok = false;
      errMsg += Form("\nERROR: Decay %i: Pull pT is biased: ptMean = %F", MOTHER_PDG_IDS[iDecay], ptMean);
    }
    if (fabs(mMean) > fabs(err_mMean)) {
      pullsok = false;
      errMsg += Form("\nERROR: Decay %i: Pull m is biased: mMean = %F", MOTHER_PDG_IDS[iDecay], mMean);
    }
    if (fabs(zMean) > fabs(err_eMean)) {
      pullsok = false;
      errMsg += Form("\nERROR: Decay %i: Pull E is biased: eMean = %F", MOTHER_PDG_IDS[iDecay], eMean);
    }

    if (xStdDev > (1 + err_xStdDev)) {
      pullsok = false;
      errMsg += Form("\nERROR: Decay %i: Error x is underestimated: xStdDev = %F", MOTHER_PDG_IDS[iDecay], xStdDev);
    }
    if (yStdDev > (1 + err_yStdDev)) {
      pullsok = false;
      errMsg += Form("\nERROR: Decay %i: Error y is underestimated: yStdDev = %F", MOTHER_PDG_IDS[iDecay], yStdDev);
    }
    if (zStdDev > (1 + err_zStdDev)) {
      pullsok = false;
      errMsg += Form("\nERROR: Decay %i: Error z is underestimated: zStdDev = %F", MOTHER_PDG_IDS[iDecay], zStdDev);
    }
    if (pxStdDev > (1 + err_pxStdDev)) {
      pullsok = false;
      errMsg += Form("\nERROR: Decay %i: Error px is underestimated: pxStdDev = %F", MOTHER_PDG_IDS[iDecay], pxStdDev);
    }
    if (pyStdDev > (1 + err_pyStdDev)) {
      pullsok = false;
      errMsg += Form("\nERROR: Decay %i: Error py is underestimated: pyStdDev = %F", MOTHER_PDG_IDS[iDecay], pyStdDev);
    }
    if (pzStdDev > (1 + err_pzStdDev)) {
      pullsok = false;
      errMsg += Form("\nERROR: Decay %i: Error pz is underestimated: pzStdDev = %F", MOTHER_PDG_IDS[iDecay], pzStdDev);
    }
    if (ptStdDev > (1 + err_ptStdDev)) {
      pullsok = false;
      errMsg += Form("\nERROR: Decay %i: Error pT is underestimated: ptStdDev = %F", MOTHER_PDG_IDS[iDecay], ptStdDev);
    }
    if (mStdDev > (1 + err_mStdDev)) {
      pullsok = false;
      errMsg += Form("\nERROR: Decay %i: Error m is underestimated: mStdDev = %F", MOTHER_PDG_IDS[iDecay], mStdDev);
    }
    if (eStdDev > (1 + err_eStdDev)) {
      pullsok = false;
      errMsg += Form("\nERROR: Decay %i: Error E is underestimated: eStdDev = %F", MOTHER_PDG_IDS[iDecay], eStdDev);
    }

    if (xStdDev < (1 - err_xStdDev)) {
      pullsok = false;
      errMsg += Form("\nERROR: Decay %i: Error x is overestimated: xStdDev = %F", MOTHER_PDG_IDS[iDecay], xStdDev);
    }
    if (yStdDev < (1 - err_yStdDev)) {
      pullsok = false;
      errMsg += Form("\nERROR: Decay %i: Error y is overestimated: yStdDev = %F", MOTHER_PDG_IDS[iDecay], yStdDev);
    }
    if (zStdDev < (1 - err_zStdDev)) {
      pullsok = false;
      errMsg += Form("\nERROR: Decay %i: Error z is overestimated: zStdDev = %F", MOTHER_PDG_IDS[iDecay], zStdDev);
    }
    if (pxStdDev < (1 - err_pxStdDev)) {
      pullsok = false;
      errMsg += Form("\nERROR: Decay %i: Error px is overestimated: pxStdDev = %F", MOTHER_PDG_IDS[iDecay], pxStdDev);
    }
    if (pyStdDev < (1 - err_pyStdDev)) {
      pullsok = false;
      errMsg += Form("\nERROR: Decay %i: Error py is overestimated: pyStdDev = %F", MOTHER_PDG_IDS[iDecay], pyStdDev);
    }
    if (pzStdDev < (1 - err_pzStdDev)) {
      pullsok = false;
      errMsg += Form("\nERROR: Decay %i: Error pz is overestimated: pzStdDev = %F", MOTHER_PDG_IDS[iDecay], pzStdDev);
    }
    if (ptStdDev < (1 - err_ptStdDev)) {
      pullsok = false;
      errMsg += Form("\nERROR: Decay %i: Error pT is overestimated: ptStdDev = %F", MOTHER_PDG_IDS[iDecay], ptStdDev);
    }
    if (mStdDev < (1 - err_mStdDev)) {
      pullsok = false;
      errMsg += Form("\nERROR: Decay %i: Error m is overestimated: mStdDev = %F", MOTHER_PDG_IDS[iDecay], mStdDev);
    }
    if (eStdDev < (1 - err_eStdDev)) {
      pullsok = false;
      errMsg += Form("\nERROR: Decay %i: Error E is overestimated: eStdDev = %F", MOTHER_PDG_IDS[iDecay], eStdDev);
    }
  }
  if (!pullsok) { cout << errMsg << endl; }

  return pullsok;
}

void PullPtDepHistos::Write()
{
  TDirectory* Pull_Dir = outputFile->mkdir("PullsVsPt");

  TDirectory* PullPtDep_Dir = Pull_Dir->mkdir("PullPtDependencies");
  std::array<TDirectory*, NUM_OF_DECAYS> DecayPullPtDep_Dir;
  std::array<TDirectory*, NUM_OF_DECAYS> MotherPullPtDep_Dir;
  std::array<std::array<TDirectory*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> DaughtersPullPtDep_Dir;
  for (int iDecay = 0; iDecay < NUM_OF_DECAYS; iDecay++) {
    DecayPullPtDep_Dir[iDecay]  = PullPtDep_Dir->mkdir(Form("%i", MOTHER_PDG_IDS[iDecay]));
    MotherPullPtDep_Dir[iDecay] = DecayPullPtDep_Dir[iDecay]->mkdir(Form("Mother_%i", MOTHER_PDG_IDS[iDecay]));
    for (int i = 0; i < NUM_OF_DAUGHTERS[iDecay]; i++) {
      DaughtersPullPtDep_Dir[iDecay][i] = DecayPullPtDep_Dir[iDecay]->mkdir(
        Form("Daughter%i_%i->%i", i + 1, MOTHER_PDG_IDS[iDecay], DAUGHTER_PDG_IDS[iDecay][i]));
    }

    MotherPullPtDep_Dir[iDecay]->cd();

    fHistPtXPullMotherLst[iDecay]->Write();
    fHistPtYPullMotherLst[iDecay]->Write();
    fHistPtZPullMotherLst[iDecay]->Write();
    fHistPtPxPullMotherLst[iDecay]->Write();
    fHistPtPyPullMotherLst[iDecay]->Write();
    fHistPtPzPullMotherLst[iDecay]->Write();
    fHistPtPtPullMotherLst[iDecay]->Write();
    fHistPtMassPullMotherLst[iDecay]->Write();
    fHistPtEPullMotherLst[iDecay]->Write();

    fHistPtXPullMotherFst[iDecay]->Write();
    fHistPtYPullMotherFst[iDecay]->Write();
    fHistPtZPullMotherFst[iDecay]->Write();
    fHistPtPxPullMotherFst[iDecay]->Write();
    fHistPtPyPullMotherFst[iDecay]->Write();
    fHistPtPzPullMotherFst[iDecay]->Write();
    fHistPtPtPullMotherFst[iDecay]->Write();
    fHistPtMassPullMotherFst[iDecay]->Write();
    fHistPtEPullMotherFst[iDecay]->Write();

    for (int i = 0; i < NUM_OF_DAUGHTERS[iDecay]; i++) {
      DaughtersPullPtDep_Dir[iDecay][i]->cd();
      fHistPtXPullDaughter[iDecay][i]->Write();
      fHistPtYPullDaughter[iDecay][i]->Write();
      fHistPtZPullDaughter[iDecay][i]->Write();
      fHistPtPxPullDaughter[iDecay][i]->Write();
      fHistPtPyPullDaughter[iDecay][i]->Write();
      fHistPtPzPullDaughter[iDecay][i]->Write();
      fHistPtPtPullDaughter[iDecay][i]->Write();
      fHistPtMassPullDaughter[iDecay][i]->Write();
      fHistPtEPullDaughter[iDecay][i]->Write();
    }
  }
  /////////////////////////////////
  /////////////////////////////////

  TDirectory* PullHistos_Dir = Pull_Dir->mkdir("PullHistos");

  std::array<TDirectory*, NUM_OF_DECAYS> DecayPullHistos_Dir;
  std::array<TDirectory*, NUM_OF_DECAYS> MotherPullHistos_Dir;
  std::array<std::array<TDirectory*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> DaughtersPullHistos_Dir;
  for (int iDecay = 0; iDecay < NUM_OF_DECAYS; iDecay++) {
    DecayPullHistos_Dir[iDecay]  = PullHistos_Dir->mkdir(Form("%i", MOTHER_PDG_IDS[iDecay]));
    MotherPullHistos_Dir[iDecay] = DecayPullHistos_Dir[iDecay]->mkdir(Form("Mother_%i", MOTHER_PDG_IDS[iDecay]));
    for (int i = 0; i < NUM_OF_DAUGHTERS[iDecay]; i++) {
      DaughtersPullHistos_Dir[iDecay][i] = DecayPullHistos_Dir[iDecay]->mkdir(
        Form("Daughter%i_%i->%i", i + 1, MOTHER_PDG_IDS[iDecay], DAUGHTER_PDG_IDS[iDecay][i]));
    }

    MotherPullHistos_Dir[iDecay]->cd();
    fHistXPullMotherLst[iDecay]->Write();
    fHistYPullMotherLst[iDecay]->Write();
    fHistZPullMotherLst[iDecay]->Write();
    fHistPxPullMotherLst[iDecay]->Write();
    fHistPyPullMotherLst[iDecay]->Write();
    fHistPzPullMotherLst[iDecay]->Write();
    fHistPtPullMotherLst[iDecay]->Write();
    fHistMassPullMotherLst[iDecay]->Write();
    fHistEPullMotherLst[iDecay]->Write();

    fHistXPullMotherFst[iDecay]->Write();
    fHistYPullMotherFst[iDecay]->Write();
    fHistZPullMotherFst[iDecay]->Write();
    fHistPxPullMotherFst[iDecay]->Write();
    fHistPyPullMotherFst[iDecay]->Write();
    fHistPzPullMotherFst[iDecay]->Write();
    fHistPtPullMotherFst[iDecay]->Write();
    fHistMassPullMotherFst[iDecay]->Write();
    fHistEPullMotherFst[iDecay]->Write();

    for (int i = 0; i < NUM_OF_DAUGHTERS[iDecay]; i++) {
      DaughtersPullHistos_Dir[iDecay][i]->cd();
      fHistXPullDaughter[iDecay][i]->Write();
      fHistYPullDaughter[iDecay][i]->Write();
      fHistZPullDaughter[iDecay][i]->Write();
      fHistPxPullDaughter[iDecay][i]->Write();
      fHistPyPullDaughter[iDecay][i]->Write();
      fHistPzPullDaughter[iDecay][i]->Write();
      fHistPtPullDaughter[iDecay][i]->Write();
      fHistMassPullDaughter[iDecay][i]->Write();
      fHistEPullDaughter[iDecay][i]->Write();
    }
  }
  /////////////////////////////////
  /////////////////////////////////
  TDirectory* MeanPullPtDepPlots_Dir = Pull_Dir->mkdir("MeanPullPtDepPlots");
  std::array<TDirectory*, NUM_OF_DECAYS> MotherMeanPullPtDepPlots_Dir;
  std::array<std::array<TDirectory*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> DaughtersMeanPullPtDepPlots_Dir;
  for (int iDecay = 0; iDecay < NUM_OF_DECAYS; iDecay++) {
    MotherMeanPullPtDepPlots_Dir[iDecay] = MeanPullPtDepPlots_Dir->mkdir(Form("Mother_%i", MOTHER_PDG_IDS[iDecay]));
    for (int i = 0; i < NUM_OF_DAUGHTERS[iDecay]; i++) {
      DaughtersMeanPullPtDepPlots_Dir[iDecay][i] = MeanPullPtDepPlots_Dir->mkdir(
        Form("Daughter%i_%i->%i", i + 1, MOTHER_PDG_IDS[iDecay], DAUGHTER_PDG_IDS[iDecay][i]));
    }

    MotherMeanPullPtDepPlots_Dir[iDecay]->cd();
    fGraphPtXMeanPullMother[iDecay]->Write();
    fGraphPtYMeanPullMother[iDecay]->Write();
    fGraphPtZMeanPullMother[iDecay]->Write();
    fGraphPtPxMeanPullMother[iDecay]->Write();
    fGraphPtPyMeanPullMother[iDecay]->Write();
    fGraphPtPzMeanPullMother[iDecay]->Write();
    fGraphPtPtMeanPullMother[iDecay]->Write();
    fGraphPtMassMeanPullMother[iDecay]->Write();
    fGraphPtEMeanPullMother[iDecay]->Write();

    for (int i = 0; i < NUM_OF_DAUGHTERS[iDecay]; i++) {
      DaughtersMeanPullPtDepPlots_Dir[iDecay][i]->cd();
      fGraphPtXMeanPullDaughter[iDecay][i]->Write();
      fGraphPtYMeanPullDaughter[iDecay][i]->Write();
      fGraphPtZMeanPullDaughter[iDecay][i]->Write();
      fGraphPtPxMeanPullDaughter[iDecay][i]->Write();
      fGraphPtPyMeanPullDaughter[iDecay][i]->Write();
      fGraphPtPzMeanPullDaughter[iDecay][i]->Write();
      fGraphPtPtMeanPullDaughter[iDecay][i]->Write();
      fGraphPtMassMeanPullDaughter[iDecay][i]->Write();
      fGraphPtEMeanPullDaughter[iDecay][i]->Write();
    }
  }
  /////////////////////////////////
  /////////////////////////////////
  TDirectory* PullWidthPtDepPlots_Dir = Pull_Dir->mkdir("PullWidthPtDepPlots");

  std::array<TDirectory*, NUM_OF_DECAYS> MotherWidthPtDepPlots_Dir;
  std::array<std::array<TDirectory*, MAX_NUM_OF_DAUGHTERS>, NUM_OF_DECAYS> DaughtersWidthPtDepPlots_Dir;
  for (int iDecay = 0; iDecay < NUM_OF_DECAYS; iDecay++) {
    MotherWidthPtDepPlots_Dir[iDecay] = PullWidthPtDepPlots_Dir->mkdir(Form("Mother_%i", MOTHER_PDG_IDS[iDecay]));
    for (int i = 0; i < NUM_OF_DAUGHTERS[iDecay]; i++) {
      DaughtersWidthPtDepPlots_Dir[iDecay][i] = PullWidthPtDepPlots_Dir->mkdir(
        Form("Daughter%i_%i->%i", i + 1, MOTHER_PDG_IDS[iDecay], DAUGHTER_PDG_IDS[iDecay][i]));
    }

    MotherWidthPtDepPlots_Dir[iDecay]->cd();
    fGraphPtXPullWidthMother[iDecay]->Write();
    fGraphPtYPullWidthMother[iDecay]->Write();
    fGraphPtZPullWidthMother[iDecay]->Write();
    fGraphPtPxPullWidthMother[iDecay]->Write();
    fGraphPtPyPullWidthMother[iDecay]->Write();
    fGraphPtPzPullWidthMother[iDecay]->Write();
    fGraphPtPtPullWidthMother[iDecay]->Write();
    fGraphPtMassPullWidthMother[iDecay]->Write();
    fGraphPtEPullWidthMother[iDecay]->Write();

    for (int i = 0; i < NUM_OF_DAUGHTERS[iDecay]; i++) {
      DaughtersWidthPtDepPlots_Dir[iDecay][i]->cd();
      fGraphPtXPullWidthDaughter[iDecay][i]->Write();
      fGraphPtYPullWidthDaughter[iDecay][i]->Write();
      fGraphPtZPullWidthDaughter[iDecay][i]->Write();
      fGraphPtPxPullWidthDaughter[iDecay][i]->Write();
      fGraphPtPyPullWidthDaughter[iDecay][i]->Write();
      fGraphPtPzPullWidthDaughter[iDecay][i]->Write();
      fGraphPtPtPullWidthDaughter[iDecay][i]->Write();
      fGraphPtMassPullWidthDaughter[iDecay][i]->Write();
      fGraphPtEPullWidthDaughter[iDecay][i]->Write();
    }
  }

  TCanvas* cPulls[NUM_OF_DECAYS];
  for (int iDecay = 0; iDecay < NUM_OF_DECAYS; iDecay++) {
    cPulls[iDecay] =
      new TCanvas(Form("Pulls_%i", MOTHER_PDG_IDS[iDecay]), Form("Pulls_%i", MOTHER_PDG_IDS[iDecay]), 1600, 1000);
    cPulls[iDecay]->Divide(3, 2);
    cPulls[iDecay]->cd(1);
    fHistXPullMotherLst[iDecay]->Draw();
    cPulls[iDecay]->cd(2);
    fHistZPullMotherLst[iDecay]->Draw();
    cPulls[iDecay]->cd(3);
    fHistPxPullMotherLst[iDecay]->Draw();
    cPulls[iDecay]->cd(4);
    fHistPtPullMotherLst[iDecay]->Draw();
    cPulls[iDecay]->cd(5);
    fHistMassPullMotherLst[iDecay]->Draw();
    cPulls[iDecay]->cd(6);
    fHistEPullMotherLst[iDecay]->Draw();
    cPulls[iDecay]->SaveAs(Form("Pulls_%i.C", MOTHER_PDG_IDS[iDecay]));
    cPulls[iDecay]->SaveAs(Form("Pulls_%i.pdf", MOTHER_PDG_IDS[iDecay]));
  }
}
