#ifdef __CLING__
#pragma cling optimize(0)
#endif
void Pulls_3122()
{
//=========Macro generated from canvas: Pulls_3122/Pulls_3122
//=========  (Mon Nov 16 15:00:19 2026) by ROOT version 6.26/10
   TCanvas *Pulls_3122 = new TCanvas("Pulls_3122", "Pulls_3122",0,0,1600,1000);
   Pulls_3122->Range(0,0,1,1);
   Pulls_3122->SetFillColor(0);
   Pulls_3122->SetBorderMode(0);
   Pulls_3122->SetBorderSize(2);
   Pulls_3122->SetFrameBorderMode(0);
  
// ------------>Primitives in pad: Pulls_3122_1
   TPad *Pulls_3122_1 = new TPad("Pulls_3122_1", "Pulls_3122_1",0.01,0.51,0.3233333,0.99);
   Pulls_3122_1->Draw();
   Pulls_3122_1->cd();
   Pulls_3122_1->Range(0,0,1,1);
   Pulls_3122_1->SetFillColor(0);
   Pulls_3122_1->SetBorderMode(0);
   Pulls_3122_1->SetBorderSize(2);
   Pulls_3122_1->SetFrameBorderMode(0);
   
   TH1D *fHistPtXPullMotherLst_3122_py__7 = new TH1D("fHistPtXPullMotherLst_3122_py__7","Pt-X Pull histo",100,-10,10);

   Int_t ci;      // for color index setting
   TColor *color; // for color definition with alpha
   ci = TColor::GetColor("#000099");
   fHistPtXPullMotherLst_3122_py__7->SetLineColor(ci);
   fHistPtXPullMotherLst_3122_py__7->GetXaxis()->SetTitle(" X Pull");
   fHistPtXPullMotherLst_3122_py__7->GetXaxis()->SetLabelFont(42);
   fHistPtXPullMotherLst_3122_py__7->GetXaxis()->SetTitleFont(42);
   fHistPtXPullMotherLst_3122_py__7->GetYaxis()->SetLabelFont(42);
   fHistPtXPullMotherLst_3122_py__7->GetYaxis()->SetTitleFont(42);
   fHistPtXPullMotherLst_3122_py__7->GetZaxis()->SetLabelFont(42);
   fHistPtXPullMotherLst_3122_py__7->GetZaxis()->SetTitleOffset(1);
   fHistPtXPullMotherLst_3122_py__7->GetZaxis()->SetTitleFont(42);
   fHistPtXPullMotherLst_3122_py__7->Draw("");
   Pulls_3122_1->Modified();
   Pulls_3122->cd();
  
// ------------>Primitives in pad: Pulls_3122_2
   TPad *Pulls_3122_2 = new TPad("Pulls_3122_2", "Pulls_3122_2",0.3433333,0.51,0.6566667,0.99);
   Pulls_3122_2->Draw();
   Pulls_3122_2->cd();
   Pulls_3122_2->Range(0,0,1,1);
   Pulls_3122_2->SetFillColor(0);
   Pulls_3122_2->SetBorderMode(0);
   Pulls_3122_2->SetBorderSize(2);
   Pulls_3122_2->SetFrameBorderMode(0);
   
   TH1D *fHistPtZPullMotherLst_3122_py__8 = new TH1D("fHistPtZPullMotherLst_3122_py__8","Pt-Z Pull histo",100,-10,10);

   ci = TColor::GetColor("#000099");
   fHistPtZPullMotherLst_3122_py__8->SetLineColor(ci);
   fHistPtZPullMotherLst_3122_py__8->GetXaxis()->SetTitle(" Z Pull");
   fHistPtZPullMotherLst_3122_py__8->GetXaxis()->SetLabelFont(42);
   fHistPtZPullMotherLst_3122_py__8->GetXaxis()->SetTitleFont(42);
   fHistPtZPullMotherLst_3122_py__8->GetYaxis()->SetLabelFont(42);
   fHistPtZPullMotherLst_3122_py__8->GetYaxis()->SetTitleFont(42);
   fHistPtZPullMotherLst_3122_py__8->GetZaxis()->SetLabelFont(42);
   fHistPtZPullMotherLst_3122_py__8->GetZaxis()->SetTitleOffset(1);
   fHistPtZPullMotherLst_3122_py__8->GetZaxis()->SetTitleFont(42);
   fHistPtZPullMotherLst_3122_py__8->Draw("");
   Pulls_3122_2->Modified();
   Pulls_3122->cd();
  
// ------------>Primitives in pad: Pulls_3122_3
   TPad *Pulls_3122_3 = new TPad("Pulls_3122_3", "Pulls_3122_3",0.6766667,0.51,0.99,0.99);
   Pulls_3122_3->Draw();
   Pulls_3122_3->cd();
   Pulls_3122_3->Range(0,0,1,1);
   Pulls_3122_3->SetFillColor(0);
   Pulls_3122_3->SetBorderMode(0);
   Pulls_3122_3->SetBorderSize(2);
   Pulls_3122_3->SetFrameBorderMode(0);
   
   TH1D *fHistPtPxPullMotherLst_3122_py__9 = new TH1D("fHistPtPxPullMotherLst_3122_py__9","Pt-Px Pull histo",50,-15,15);

   ci = TColor::GetColor("#000099");
   fHistPtPxPullMotherLst_3122_py__9->SetLineColor(ci);
   fHistPtPxPullMotherLst_3122_py__9->GetXaxis()->SetTitle(" Px Pull");
   fHistPtPxPullMotherLst_3122_py__9->GetXaxis()->SetLabelFont(42);
   fHistPtPxPullMotherLst_3122_py__9->GetXaxis()->SetTitleFont(42);
   fHistPtPxPullMotherLst_3122_py__9->GetYaxis()->SetLabelFont(42);
   fHistPtPxPullMotherLst_3122_py__9->GetYaxis()->SetTitleFont(42);
   fHistPtPxPullMotherLst_3122_py__9->GetZaxis()->SetLabelFont(42);
   fHistPtPxPullMotherLst_3122_py__9->GetZaxis()->SetTitleOffset(1);
   fHistPtPxPullMotherLst_3122_py__9->GetZaxis()->SetTitleFont(42);
   fHistPtPxPullMotherLst_3122_py__9->Draw("");
   Pulls_3122_3->Modified();
   Pulls_3122->cd();
  
// ------------>Primitives in pad: Pulls_3122_4
   TPad *Pulls_3122_4 = new TPad("Pulls_3122_4", "Pulls_3122_4",0.01,0.01,0.3233333,0.49);
   Pulls_3122_4->Draw();
   Pulls_3122_4->cd();
   Pulls_3122_4->Range(0,0,1,1);
   Pulls_3122_4->SetFillColor(0);
   Pulls_3122_4->SetBorderMode(0);
   Pulls_3122_4->SetBorderSize(2);
   Pulls_3122_4->SetFrameBorderMode(0);
   
   TH1D *fHistPtPtPullMotherLst_3122_py__10 = new TH1D("fHistPtPtPullMotherLst_3122_py__10","Pt-Pt Pull histo",50,-15,15);

   ci = TColor::GetColor("#000099");
   fHistPtPtPullMotherLst_3122_py__10->SetLineColor(ci);
   fHistPtPtPullMotherLst_3122_py__10->GetXaxis()->SetTitle(" Pt Pull");
   fHistPtPtPullMotherLst_3122_py__10->GetXaxis()->SetLabelFont(42);
   fHistPtPtPullMotherLst_3122_py__10->GetXaxis()->SetTitleFont(42);
   fHistPtPtPullMotherLst_3122_py__10->GetYaxis()->SetLabelFont(42);
   fHistPtPtPullMotherLst_3122_py__10->GetYaxis()->SetTitleFont(42);
   fHistPtPtPullMotherLst_3122_py__10->GetZaxis()->SetLabelFont(42);
   fHistPtPtPullMotherLst_3122_py__10->GetZaxis()->SetTitleOffset(1);
   fHistPtPtPullMotherLst_3122_py__10->GetZaxis()->SetTitleFont(42);
   fHistPtPtPullMotherLst_3122_py__10->Draw("");
   Pulls_3122_4->Modified();
   Pulls_3122->cd();
  
// ------------>Primitives in pad: Pulls_3122_5
   TPad *Pulls_3122_5 = new TPad("Pulls_3122_5", "Pulls_3122_5",0.3433333,0.01,0.6566667,0.49);
   Pulls_3122_5->Draw();
   Pulls_3122_5->cd();
   Pulls_3122_5->Range(0,0,1,1);
   Pulls_3122_5->SetFillColor(0);
   Pulls_3122_5->SetBorderMode(0);
   Pulls_3122_5->SetBorderSize(2);
   Pulls_3122_5->SetFrameBorderMode(0);
   
   TH1D *fHistPtMassPullMotherLst_3122_py__11 = new TH1D("fHistPtMassPullMotherLst_3122_py__11","Pt-Mass Pull histo",50,-5,5);

   ci = TColor::GetColor("#000099");
   fHistPtMassPullMotherLst_3122_py__11->SetLineColor(ci);
   fHistPtMassPullMotherLst_3122_py__11->GetXaxis()->SetTitle(" Mass Pull");
   fHistPtMassPullMotherLst_3122_py__11->GetXaxis()->SetLabelFont(42);
   fHistPtMassPullMotherLst_3122_py__11->GetXaxis()->SetTitleFont(42);
   fHistPtMassPullMotherLst_3122_py__11->GetYaxis()->SetLabelFont(42);
   fHistPtMassPullMotherLst_3122_py__11->GetYaxis()->SetTitleFont(42);
   fHistPtMassPullMotherLst_3122_py__11->GetZaxis()->SetLabelFont(42);
   fHistPtMassPullMotherLst_3122_py__11->GetZaxis()->SetTitleOffset(1);
   fHistPtMassPullMotherLst_3122_py__11->GetZaxis()->SetTitleFont(42);
   fHistPtMassPullMotherLst_3122_py__11->Draw("");
   Pulls_3122_5->Modified();
   Pulls_3122->cd();
  
// ------------>Primitives in pad: Pulls_3122_6
   TPad *Pulls_3122_6 = new TPad("Pulls_3122_6", "Pulls_3122_6",0.6766667,0.01,0.99,0.49);
   Pulls_3122_6->Draw();
   Pulls_3122_6->cd();
   Pulls_3122_6->Range(0,0,1,1);
   Pulls_3122_6->SetFillColor(0);
   Pulls_3122_6->SetBorderMode(0);
   Pulls_3122_6->SetBorderSize(2);
   Pulls_3122_6->SetFrameBorderMode(0);
   
   TH1D *fHistPtEPullMotherLst_3122_py__12 = new TH1D("fHistPtEPullMotherLst_3122_py__12","Pt-E Pull histo",50,-15,15);

   ci = TColor::GetColor("#000099");
   fHistPtEPullMotherLst_3122_py__12->SetLineColor(ci);
   fHistPtEPullMotherLst_3122_py__12->GetXaxis()->SetTitle(" E Pull");
   fHistPtEPullMotherLst_3122_py__12->GetXaxis()->SetLabelFont(42);
   fHistPtEPullMotherLst_3122_py__12->GetXaxis()->SetTitleFont(42);
   fHistPtEPullMotherLst_3122_py__12->GetYaxis()->SetLabelFont(42);
   fHistPtEPullMotherLst_3122_py__12->GetYaxis()->SetTitleFont(42);
   fHistPtEPullMotherLst_3122_py__12->GetZaxis()->SetLabelFont(42);
   fHistPtEPullMotherLst_3122_py__12->GetZaxis()->SetTitleOffset(1);
   fHistPtEPullMotherLst_3122_py__12->GetZaxis()->SetTitleFont(42);
   fHistPtEPullMotherLst_3122_py__12->Draw("");
   Pulls_3122_6->Modified();
   Pulls_3122->cd();
   Pulls_3122->Modified();
   Pulls_3122->cd();
   Pulls_3122->SetSelected(Pulls_3122);
}
