#ifdef __CLING__
#pragma cling optimize(0)
#endif
void Resids_3122()
{
//=========Macro generated from canvas: Resids_3122/Resids_3122
//=========  (Mon Nov 16 15:00:19 2026) by ROOT version 6.26/10
   TCanvas *Resids_3122 = new TCanvas("Resids_3122", "Resids_3122",0,0,1600,1000);
   Resids_3122->Range(0,0,1,1);
   Resids_3122->SetFillColor(0);
   Resids_3122->SetBorderMode(0);
   Resids_3122->SetBorderSize(2);
   Resids_3122->SetFrameBorderMode(0);
  
// ------------>Primitives in pad: Resids_3122_1
   TPad *Resids_3122_1 = new TPad("Resids_3122_1", "Resids_3122_1",0.01,0.51,0.3233333,0.99);
   Resids_3122_1->Draw();
   Resids_3122_1->cd();
   Resids_3122_1->Range(0,0,1,1);
   Resids_3122_1->SetFillColor(0);
   Resids_3122_1->SetBorderMode(0);
   Resids_3122_1->SetBorderSize(2);
   Resids_3122_1->SetFrameBorderMode(0);
   
   TH1D *fHistPtXResidMotherLst_3122_py__19 = new TH1D("fHistPtXResidMotherLst_3122_py__19","Pt-X Resid histo",100,-0.1,0.1);

   Int_t ci;      // for color index setting
   TColor *color; // for color definition with alpha
   ci = TColor::GetColor("#000099");
   fHistPtXResidMotherLst_3122_py__19->SetLineColor(ci);
   fHistPtXResidMotherLst_3122_py__19->GetXaxis()->SetTitle(" X residual, cm");
   fHistPtXResidMotherLst_3122_py__19->GetXaxis()->SetLabelFont(42);
   fHistPtXResidMotherLst_3122_py__19->GetXaxis()->SetTitleFont(42);
   fHistPtXResidMotherLst_3122_py__19->GetYaxis()->SetLabelFont(42);
   fHistPtXResidMotherLst_3122_py__19->GetYaxis()->SetTitleFont(42);
   fHistPtXResidMotherLst_3122_py__19->GetZaxis()->SetLabelFont(42);
   fHistPtXResidMotherLst_3122_py__19->GetZaxis()->SetTitleOffset(1);
   fHistPtXResidMotherLst_3122_py__19->GetZaxis()->SetTitleFont(42);
   fHistPtXResidMotherLst_3122_py__19->Draw("");
   Resids_3122_1->Modified();
   Resids_3122->cd();
  
// ------------>Primitives in pad: Resids_3122_2
   TPad *Resids_3122_2 = new TPad("Resids_3122_2", "Resids_3122_2",0.3433333,0.51,0.6566667,0.99);
   Resids_3122_2->Draw();
   Resids_3122_2->cd();
   Resids_3122_2->Range(0,0,1,1);
   Resids_3122_2->SetFillColor(0);
   Resids_3122_2->SetBorderMode(0);
   Resids_3122_2->SetBorderSize(2);
   Resids_3122_2->SetFrameBorderMode(0);
   
   TH1D *fHistPtZResidMotherLst_3122_py__20 = new TH1D("fHistPtZResidMotherLst_3122_py__20","Pt-Z Resid histo",100,-0.1,0.1);

   ci = TColor::GetColor("#000099");
   fHistPtZResidMotherLst_3122_py__20->SetLineColor(ci);
   fHistPtZResidMotherLst_3122_py__20->GetXaxis()->SetTitle(" Z residual, cm");
   fHistPtZResidMotherLst_3122_py__20->GetXaxis()->SetLabelFont(42);
   fHistPtZResidMotherLst_3122_py__20->GetXaxis()->SetTitleFont(42);
   fHistPtZResidMotherLst_3122_py__20->GetYaxis()->SetLabelFont(42);
   fHistPtZResidMotherLst_3122_py__20->GetYaxis()->SetTitleFont(42);
   fHistPtZResidMotherLst_3122_py__20->GetZaxis()->SetLabelFont(42);
   fHistPtZResidMotherLst_3122_py__20->GetZaxis()->SetTitleOffset(1);
   fHistPtZResidMotherLst_3122_py__20->GetZaxis()->SetTitleFont(42);
   fHistPtZResidMotherLst_3122_py__20->Draw("");
   Resids_3122_2->Modified();
   Resids_3122->cd();
  
// ------------>Primitives in pad: Resids_3122_3
   TPad *Resids_3122_3 = new TPad("Resids_3122_3", "Resids_3122_3",0.6766667,0.51,0.99,0.99);
   Resids_3122_3->Draw();
   Resids_3122_3->cd();
   Resids_3122_3->Range(0,0,1,1);
   Resids_3122_3->SetFillColor(0);
   Resids_3122_3->SetBorderMode(0);
   Resids_3122_3->SetBorderSize(2);
   Resids_3122_3->SetFrameBorderMode(0);
   
   TH1D *fHistPtPxResidMotherLst_3122_py__21 = new TH1D("fHistPtPxResidMotherLst_3122_py__21","Pt-Px Resid histo",50,-0.15,0.15);

   ci = TColor::GetColor("#000099");
   fHistPtPxResidMotherLst_3122_py__21->SetLineColor(ci);
   fHistPtPxResidMotherLst_3122_py__21->GetXaxis()->SetTitle(" Px residual, GeV");
   fHistPtPxResidMotherLst_3122_py__21->GetXaxis()->SetLabelFont(42);
   fHistPtPxResidMotherLst_3122_py__21->GetXaxis()->SetTitleFont(42);
   fHistPtPxResidMotherLst_3122_py__21->GetYaxis()->SetLabelFont(42);
   fHistPtPxResidMotherLst_3122_py__21->GetYaxis()->SetTitleFont(42);
   fHistPtPxResidMotherLst_3122_py__21->GetZaxis()->SetLabelFont(42);
   fHistPtPxResidMotherLst_3122_py__21->GetZaxis()->SetTitleOffset(1);
   fHistPtPxResidMotherLst_3122_py__21->GetZaxis()->SetTitleFont(42);
   fHistPtPxResidMotherLst_3122_py__21->Draw("");
   Resids_3122_3->Modified();
   Resids_3122->cd();
  
// ------------>Primitives in pad: Resids_3122_4
   TPad *Resids_3122_4 = new TPad("Resids_3122_4", "Resids_3122_4",0.01,0.01,0.3233333,0.49);
   Resids_3122_4->Draw();
   Resids_3122_4->cd();
   Resids_3122_4->Range(0,0,1,1);
   Resids_3122_4->SetFillColor(0);
   Resids_3122_4->SetBorderMode(0);
   Resids_3122_4->SetBorderSize(2);
   Resids_3122_4->SetFrameBorderMode(0);
   
   TH1D *fHistPtPtResidMotherLst_3122_py__22 = new TH1D("fHistPtPtResidMotherLst_3122_py__22","Pt-Pt Resid histo",50,-0.15,0.15);

   ci = TColor::GetColor("#000099");
   fHistPtPtResidMotherLst_3122_py__22->SetLineColor(ci);
   fHistPtPtResidMotherLst_3122_py__22->GetXaxis()->SetTitle(" Pt residual, GeV");
   fHistPtPtResidMotherLst_3122_py__22->GetXaxis()->SetLabelFont(42);
   fHistPtPtResidMotherLst_3122_py__22->GetXaxis()->SetTitleFont(42);
   fHistPtPtResidMotherLst_3122_py__22->GetYaxis()->SetLabelFont(42);
   fHistPtPtResidMotherLst_3122_py__22->GetYaxis()->SetTitleFont(42);
   fHistPtPtResidMotherLst_3122_py__22->GetZaxis()->SetLabelFont(42);
   fHistPtPtResidMotherLst_3122_py__22->GetZaxis()->SetTitleOffset(1);
   fHistPtPtResidMotherLst_3122_py__22->GetZaxis()->SetTitleFont(42);
   fHistPtPtResidMotherLst_3122_py__22->Draw("");
   Resids_3122_4->Modified();
   Resids_3122->cd();
  
// ------------>Primitives in pad: Resids_3122_5
   TPad *Resids_3122_5 = new TPad("Resids_3122_5", "Resids_3122_5",0.3433333,0.01,0.6566667,0.49);
   Resids_3122_5->Draw();
   Resids_3122_5->cd();
   Resids_3122_5->Range(0,0,1,1);
   Resids_3122_5->SetFillColor(0);
   Resids_3122_5->SetBorderMode(0);
   Resids_3122_5->SetBorderSize(2);
   Resids_3122_5->SetFrameBorderMode(0);
   
   TH1D *fHistPtMassResidMotherLst_3122_py__23 = new TH1D("fHistPtMassResidMotherLst_3122_py__23","Pt-Mass Resid histo",50,-0.4,0.4);

   ci = TColor::GetColor("#000099");
   fHistPtMassResidMotherLst_3122_py__23->SetLineColor(ci);
   fHistPtMassResidMotherLst_3122_py__23->GetXaxis()->SetTitle(" Mass residual, GeV");
   fHistPtMassResidMotherLst_3122_py__23->GetXaxis()->SetLabelFont(42);
   fHistPtMassResidMotherLst_3122_py__23->GetXaxis()->SetTitleFont(42);
   fHistPtMassResidMotherLst_3122_py__23->GetYaxis()->SetLabelFont(42);
   fHistPtMassResidMotherLst_3122_py__23->GetYaxis()->SetTitleFont(42);
   fHistPtMassResidMotherLst_3122_py__23->GetZaxis()->SetLabelFont(42);
   fHistPtMassResidMotherLst_3122_py__23->GetZaxis()->SetTitleOffset(1);
   fHistPtMassResidMotherLst_3122_py__23->GetZaxis()->SetTitleFont(42);
   fHistPtMassResidMotherLst_3122_py__23->Draw("");
   Resids_3122_5->Modified();
   Resids_3122->cd();
  
// ------------>Primitives in pad: Resids_3122_6
   TPad *Resids_3122_6 = new TPad("Resids_3122_6", "Resids_3122_6",0.6766667,0.01,0.99,0.49);
   Resids_3122_6->Draw();
   Resids_3122_6->cd();
   Resids_3122_6->Range(0,0,1,1);
   Resids_3122_6->SetFillColor(0);
   Resids_3122_6->SetBorderMode(0);
   Resids_3122_6->SetBorderSize(2);
   Resids_3122_6->SetFrameBorderMode(0);
   
   TH1D *fHistPtEResidMotherLst_3122_py__24 = new TH1D("fHistPtEResidMotherLst_3122_py__24","Pt-E Resid histo",50,-0.15,0.15);

   ci = TColor::GetColor("#000099");
   fHistPtEResidMotherLst_3122_py__24->SetLineColor(ci);
   fHistPtEResidMotherLst_3122_py__24->GetXaxis()->SetTitle(" E residual, GeV");
   fHistPtEResidMotherLst_3122_py__24->GetXaxis()->SetLabelFont(42);
   fHistPtEResidMotherLst_3122_py__24->GetXaxis()->SetTitleFont(42);
   fHistPtEResidMotherLst_3122_py__24->GetYaxis()->SetLabelFont(42);
   fHistPtEResidMotherLst_3122_py__24->GetYaxis()->SetTitleFont(42);
   fHistPtEResidMotherLst_3122_py__24->GetZaxis()->SetLabelFont(42);
   fHistPtEResidMotherLst_3122_py__24->GetZaxis()->SetTitleOffset(1);
   fHistPtEResidMotherLst_3122_py__24->GetZaxis()->SetTitleFont(42);
   fHistPtEResidMotherLst_3122_py__24->Draw("");
   Resids_3122_6->Modified();
   Resids_3122->cd();
   Resids_3122->Modified();
   Resids_3122->cd();
   Resids_3122->SetSelected(Resids_3122);
}
