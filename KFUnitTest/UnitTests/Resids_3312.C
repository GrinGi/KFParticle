#ifdef __CLING__
#pragma cling optimize(0)
#endif
void Resids_3312()
{
//=========Macro generated from canvas: Resids_3312/Resids_3312
//=========  (Mon Nov 16 15:00:19 2026) by ROOT version 6.26/10
   TCanvas *Resids_3312 = new TCanvas("Resids_3312", "Resids_3312",0,0,1600,1000);
   Resids_3312->Range(0,0,1,1);
   Resids_3312->SetFillColor(0);
   Resids_3312->SetBorderMode(0);
   Resids_3312->SetBorderSize(2);
   Resids_3312->SetFrameBorderMode(0);
  
// ------------>Primitives in pad: Resids_3312_1
   TPad *Resids_3312_1 = new TPad("Resids_3312_1", "Resids_3312_1",0.01,0.51,0.3233333,0.99);
   Resids_3312_1->Draw();
   Resids_3312_1->cd();
   Resids_3312_1->Range(0,0,1,1);
   Resids_3312_1->SetFillColor(0);
   Resids_3312_1->SetBorderMode(0);
   Resids_3312_1->SetBorderSize(2);
   Resids_3312_1->SetFrameBorderMode(0);
   
   TH1D *fHistPtXResidMotherLst_3312_py__13 = new TH1D("fHistPtXResidMotherLst_3312_py__13","Pt-X Resid histo",100,-0.1,0.1);

   Int_t ci;      // for color index setting
   TColor *color; // for color definition with alpha
   ci = TColor::GetColor("#000099");
   fHistPtXResidMotherLst_3312_py__13->SetLineColor(ci);
   fHistPtXResidMotherLst_3312_py__13->GetXaxis()->SetTitle(" X residual, cm");
   fHistPtXResidMotherLst_3312_py__13->GetXaxis()->SetLabelFont(42);
   fHistPtXResidMotherLst_3312_py__13->GetXaxis()->SetTitleFont(42);
   fHistPtXResidMotherLst_3312_py__13->GetYaxis()->SetLabelFont(42);
   fHistPtXResidMotherLst_3312_py__13->GetYaxis()->SetTitleFont(42);
   fHistPtXResidMotherLst_3312_py__13->GetZaxis()->SetLabelFont(42);
   fHistPtXResidMotherLst_3312_py__13->GetZaxis()->SetTitleOffset(1);
   fHistPtXResidMotherLst_3312_py__13->GetZaxis()->SetTitleFont(42);
   fHistPtXResidMotherLst_3312_py__13->Draw("");
   Resids_3312_1->Modified();
   Resids_3312->cd();
  
// ------------>Primitives in pad: Resids_3312_2
   TPad *Resids_3312_2 = new TPad("Resids_3312_2", "Resids_3312_2",0.3433333,0.51,0.6566667,0.99);
   Resids_3312_2->Draw();
   Resids_3312_2->cd();
   Resids_3312_2->Range(0,0,1,1);
   Resids_3312_2->SetFillColor(0);
   Resids_3312_2->SetBorderMode(0);
   Resids_3312_2->SetBorderSize(2);
   Resids_3312_2->SetFrameBorderMode(0);
   
   TH1D *fHistPtZResidMotherLst_3312_py__14 = new TH1D("fHistPtZResidMotherLst_3312_py__14","Pt-Z Resid histo",100,-0.1,0.1);

   ci = TColor::GetColor("#000099");
   fHistPtZResidMotherLst_3312_py__14->SetLineColor(ci);
   fHistPtZResidMotherLst_3312_py__14->GetXaxis()->SetTitle(" Z residual, cm");
   fHistPtZResidMotherLst_3312_py__14->GetXaxis()->SetLabelFont(42);
   fHistPtZResidMotherLst_3312_py__14->GetXaxis()->SetTitleFont(42);
   fHistPtZResidMotherLst_3312_py__14->GetYaxis()->SetLabelFont(42);
   fHistPtZResidMotherLst_3312_py__14->GetYaxis()->SetTitleFont(42);
   fHistPtZResidMotherLst_3312_py__14->GetZaxis()->SetLabelFont(42);
   fHistPtZResidMotherLst_3312_py__14->GetZaxis()->SetTitleOffset(1);
   fHistPtZResidMotherLst_3312_py__14->GetZaxis()->SetTitleFont(42);
   fHistPtZResidMotherLst_3312_py__14->Draw("");
   Resids_3312_2->Modified();
   Resids_3312->cd();
  
// ------------>Primitives in pad: Resids_3312_3
   TPad *Resids_3312_3 = new TPad("Resids_3312_3", "Resids_3312_3",0.6766667,0.51,0.99,0.99);
   Resids_3312_3->Draw();
   Resids_3312_3->cd();
   Resids_3312_3->Range(0,0,1,1);
   Resids_3312_3->SetFillColor(0);
   Resids_3312_3->SetBorderMode(0);
   Resids_3312_3->SetBorderSize(2);
   Resids_3312_3->SetFrameBorderMode(0);
   
   TH1D *fHistPtPxResidMotherLst_3312_py__15 = new TH1D("fHistPtPxResidMotherLst_3312_py__15","Pt-Px Resid histo",50,-0.15,0.15);

   ci = TColor::GetColor("#000099");
   fHistPtPxResidMotherLst_3312_py__15->SetLineColor(ci);
   fHistPtPxResidMotherLst_3312_py__15->GetXaxis()->SetTitle(" Px residual, GeV");
   fHistPtPxResidMotherLst_3312_py__15->GetXaxis()->SetLabelFont(42);
   fHistPtPxResidMotherLst_3312_py__15->GetXaxis()->SetTitleFont(42);
   fHistPtPxResidMotherLst_3312_py__15->GetYaxis()->SetLabelFont(42);
   fHistPtPxResidMotherLst_3312_py__15->GetYaxis()->SetTitleFont(42);
   fHistPtPxResidMotherLst_3312_py__15->GetZaxis()->SetLabelFont(42);
   fHistPtPxResidMotherLst_3312_py__15->GetZaxis()->SetTitleOffset(1);
   fHistPtPxResidMotherLst_3312_py__15->GetZaxis()->SetTitleFont(42);
   fHistPtPxResidMotherLst_3312_py__15->Draw("");
   Resids_3312_3->Modified();
   Resids_3312->cd();
  
// ------------>Primitives in pad: Resids_3312_4
   TPad *Resids_3312_4 = new TPad("Resids_3312_4", "Resids_3312_4",0.01,0.01,0.3233333,0.49);
   Resids_3312_4->Draw();
   Resids_3312_4->cd();
   Resids_3312_4->Range(0,0,1,1);
   Resids_3312_4->SetFillColor(0);
   Resids_3312_4->SetBorderMode(0);
   Resids_3312_4->SetBorderSize(2);
   Resids_3312_4->SetFrameBorderMode(0);
   
   TH1D *fHistPtPtResidMotherLst_3312_py__16 = new TH1D("fHistPtPtResidMotherLst_3312_py__16","Pt-Pt Resid histo",50,-0.15,0.15);

   ci = TColor::GetColor("#000099");
   fHistPtPtResidMotherLst_3312_py__16->SetLineColor(ci);
   fHistPtPtResidMotherLst_3312_py__16->GetXaxis()->SetTitle(" Pt residual, GeV");
   fHistPtPtResidMotherLst_3312_py__16->GetXaxis()->SetLabelFont(42);
   fHistPtPtResidMotherLst_3312_py__16->GetXaxis()->SetTitleFont(42);
   fHistPtPtResidMotherLst_3312_py__16->GetYaxis()->SetLabelFont(42);
   fHistPtPtResidMotherLst_3312_py__16->GetYaxis()->SetTitleFont(42);
   fHistPtPtResidMotherLst_3312_py__16->GetZaxis()->SetLabelFont(42);
   fHistPtPtResidMotherLst_3312_py__16->GetZaxis()->SetTitleOffset(1);
   fHistPtPtResidMotherLst_3312_py__16->GetZaxis()->SetTitleFont(42);
   fHistPtPtResidMotherLst_3312_py__16->Draw("");
   Resids_3312_4->Modified();
   Resids_3312->cd();
  
// ------------>Primitives in pad: Resids_3312_5
   TPad *Resids_3312_5 = new TPad("Resids_3312_5", "Resids_3312_5",0.3433333,0.01,0.6566667,0.49);
   Resids_3312_5->Draw();
   Resids_3312_5->cd();
   Resids_3312_5->Range(0,0,1,1);
   Resids_3312_5->SetFillColor(0);
   Resids_3312_5->SetBorderMode(0);
   Resids_3312_5->SetBorderSize(2);
   Resids_3312_5->SetFrameBorderMode(0);
   
   TH1D *fHistPtMassResidMotherLst_3312_py__17 = new TH1D("fHistPtMassResidMotherLst_3312_py__17","Pt-Mass Resid histo",50,-0.4,0.4);

   ci = TColor::GetColor("#000099");
   fHistPtMassResidMotherLst_3312_py__17->SetLineColor(ci);
   fHistPtMassResidMotherLst_3312_py__17->GetXaxis()->SetTitle(" Mass residual, GeV");
   fHistPtMassResidMotherLst_3312_py__17->GetXaxis()->SetLabelFont(42);
   fHistPtMassResidMotherLst_3312_py__17->GetXaxis()->SetTitleFont(42);
   fHistPtMassResidMotherLst_3312_py__17->GetYaxis()->SetLabelFont(42);
   fHistPtMassResidMotherLst_3312_py__17->GetYaxis()->SetTitleFont(42);
   fHistPtMassResidMotherLst_3312_py__17->GetZaxis()->SetLabelFont(42);
   fHistPtMassResidMotherLst_3312_py__17->GetZaxis()->SetTitleOffset(1);
   fHistPtMassResidMotherLst_3312_py__17->GetZaxis()->SetTitleFont(42);
   fHistPtMassResidMotherLst_3312_py__17->Draw("");
   Resids_3312_5->Modified();
   Resids_3312->cd();
  
// ------------>Primitives in pad: Resids_3312_6
   TPad *Resids_3312_6 = new TPad("Resids_3312_6", "Resids_3312_6",0.6766667,0.01,0.99,0.49);
   Resids_3312_6->Draw();
   Resids_3312_6->cd();
   Resids_3312_6->Range(0,0,1,1);
   Resids_3312_6->SetFillColor(0);
   Resids_3312_6->SetBorderMode(0);
   Resids_3312_6->SetBorderSize(2);
   Resids_3312_6->SetFrameBorderMode(0);
   
   TH1D *fHistPtEResidMotherLst_3312_py__18 = new TH1D("fHistPtEResidMotherLst_3312_py__18","Pt-E Resid histo",50,-0.15,0.15);

   ci = TColor::GetColor("#000099");
   fHistPtEResidMotherLst_3312_py__18->SetLineColor(ci);
   fHistPtEResidMotherLst_3312_py__18->GetXaxis()->SetTitle(" E residual, GeV");
   fHistPtEResidMotherLst_3312_py__18->GetXaxis()->SetLabelFont(42);
   fHistPtEResidMotherLst_3312_py__18->GetXaxis()->SetTitleFont(42);
   fHistPtEResidMotherLst_3312_py__18->GetYaxis()->SetLabelFont(42);
   fHistPtEResidMotherLst_3312_py__18->GetYaxis()->SetTitleFont(42);
   fHistPtEResidMotherLst_3312_py__18->GetZaxis()->SetLabelFont(42);
   fHistPtEResidMotherLst_3312_py__18->GetZaxis()->SetTitleOffset(1);
   fHistPtEResidMotherLst_3312_py__18->GetZaxis()->SetTitleFont(42);
   fHistPtEResidMotherLst_3312_py__18->Draw("");
   Resids_3312_6->Modified();
   Resids_3312->cd();
   Resids_3312->Modified();
   Resids_3312->cd();
   Resids_3312->SetSelected(Resids_3312);
}
