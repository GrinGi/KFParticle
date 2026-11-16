#ifdef __CLING__
#pragma cling optimize(0)
#endif
void Pulls_3312()
{
//=========Macro generated from canvas: Pulls_3312/Pulls_3312
//=========  (Mon Nov 16 15:00:19 2026) by ROOT version 6.26/10
   TCanvas *Pulls_3312 = new TCanvas("Pulls_3312", "Pulls_3312",0,0,1600,1000);
   Pulls_3312->Range(0,0,1,1);
   Pulls_3312->SetFillColor(0);
   Pulls_3312->SetBorderMode(0);
   Pulls_3312->SetBorderSize(2);
   Pulls_3312->SetFrameBorderMode(0);
  
// ------------>Primitives in pad: Pulls_3312_1
   TPad *Pulls_3312_1 = new TPad("Pulls_3312_1", "Pulls_3312_1",0.01,0.51,0.3233333,0.99);
   Pulls_3312_1->Draw();
   Pulls_3312_1->cd();
   Pulls_3312_1->Range(0,0,1,1);
   Pulls_3312_1->SetFillColor(0);
   Pulls_3312_1->SetBorderMode(0);
   Pulls_3312_1->SetBorderSize(2);
   Pulls_3312_1->SetFrameBorderMode(0);
   
   TH1D *fHistPtXPullMotherLst_3312_py__1 = new TH1D("fHistPtXPullMotherLst_3312_py__1","Pt-X Pull histo",100,-10,10);

   Int_t ci;      // for color index setting
   TColor *color; // for color definition with alpha
   ci = TColor::GetColor("#000099");
   fHistPtXPullMotherLst_3312_py__1->SetLineColor(ci);
   fHistPtXPullMotherLst_3312_py__1->GetXaxis()->SetTitle(" X Pull");
   fHistPtXPullMotherLst_3312_py__1->GetXaxis()->SetLabelFont(42);
   fHistPtXPullMotherLst_3312_py__1->GetXaxis()->SetTitleFont(42);
   fHistPtXPullMotherLst_3312_py__1->GetYaxis()->SetLabelFont(42);
   fHistPtXPullMotherLst_3312_py__1->GetYaxis()->SetTitleFont(42);
   fHistPtXPullMotherLst_3312_py__1->GetZaxis()->SetLabelFont(42);
   fHistPtXPullMotherLst_3312_py__1->GetZaxis()->SetTitleOffset(1);
   fHistPtXPullMotherLst_3312_py__1->GetZaxis()->SetTitleFont(42);
   fHistPtXPullMotherLst_3312_py__1->Draw("");
   Pulls_3312_1->Modified();
   Pulls_3312->cd();
  
// ------------>Primitives in pad: Pulls_3312_2
   TPad *Pulls_3312_2 = new TPad("Pulls_3312_2", "Pulls_3312_2",0.3433333,0.51,0.6566667,0.99);
   Pulls_3312_2->Draw();
   Pulls_3312_2->cd();
   Pulls_3312_2->Range(0,0,1,1);
   Pulls_3312_2->SetFillColor(0);
   Pulls_3312_2->SetBorderMode(0);
   Pulls_3312_2->SetBorderSize(2);
   Pulls_3312_2->SetFrameBorderMode(0);
   
   TH1D *fHistPtZPullMotherLst_3312_py__2 = new TH1D("fHistPtZPullMotherLst_3312_py__2","Pt-Z Pull histo",100,-10,10);

   ci = TColor::GetColor("#000099");
   fHistPtZPullMotherLst_3312_py__2->SetLineColor(ci);
   fHistPtZPullMotherLst_3312_py__2->GetXaxis()->SetTitle(" Z Pull");
   fHistPtZPullMotherLst_3312_py__2->GetXaxis()->SetLabelFont(42);
   fHistPtZPullMotherLst_3312_py__2->GetXaxis()->SetTitleFont(42);
   fHistPtZPullMotherLst_3312_py__2->GetYaxis()->SetLabelFont(42);
   fHistPtZPullMotherLst_3312_py__2->GetYaxis()->SetTitleFont(42);
   fHistPtZPullMotherLst_3312_py__2->GetZaxis()->SetLabelFont(42);
   fHistPtZPullMotherLst_3312_py__2->GetZaxis()->SetTitleOffset(1);
   fHistPtZPullMotherLst_3312_py__2->GetZaxis()->SetTitleFont(42);
   fHistPtZPullMotherLst_3312_py__2->Draw("");
   Pulls_3312_2->Modified();
   Pulls_3312->cd();
  
// ------------>Primitives in pad: Pulls_3312_3
   TPad *Pulls_3312_3 = new TPad("Pulls_3312_3", "Pulls_3312_3",0.6766667,0.51,0.99,0.99);
   Pulls_3312_3->Draw();
   Pulls_3312_3->cd();
   Pulls_3312_3->Range(0,0,1,1);
   Pulls_3312_3->SetFillColor(0);
   Pulls_3312_3->SetBorderMode(0);
   Pulls_3312_3->SetBorderSize(2);
   Pulls_3312_3->SetFrameBorderMode(0);
   
   TH1D *fHistPtPxPullMotherLst_3312_py__3 = new TH1D("fHistPtPxPullMotherLst_3312_py__3","Pt-Px Pull histo",50,-15,15);

   ci = TColor::GetColor("#000099");
   fHistPtPxPullMotherLst_3312_py__3->SetLineColor(ci);
   fHistPtPxPullMotherLst_3312_py__3->GetXaxis()->SetTitle(" Px Pull");
   fHistPtPxPullMotherLst_3312_py__3->GetXaxis()->SetLabelFont(42);
   fHistPtPxPullMotherLst_3312_py__3->GetXaxis()->SetTitleFont(42);
   fHistPtPxPullMotherLst_3312_py__3->GetYaxis()->SetLabelFont(42);
   fHistPtPxPullMotherLst_3312_py__3->GetYaxis()->SetTitleFont(42);
   fHistPtPxPullMotherLst_3312_py__3->GetZaxis()->SetLabelFont(42);
   fHistPtPxPullMotherLst_3312_py__3->GetZaxis()->SetTitleOffset(1);
   fHistPtPxPullMotherLst_3312_py__3->GetZaxis()->SetTitleFont(42);
   fHistPtPxPullMotherLst_3312_py__3->Draw("");
   Pulls_3312_3->Modified();
   Pulls_3312->cd();
  
// ------------>Primitives in pad: Pulls_3312_4
   TPad *Pulls_3312_4 = new TPad("Pulls_3312_4", "Pulls_3312_4",0.01,0.01,0.3233333,0.49);
   Pulls_3312_4->Draw();
   Pulls_3312_4->cd();
   Pulls_3312_4->Range(0,0,1,1);
   Pulls_3312_4->SetFillColor(0);
   Pulls_3312_4->SetBorderMode(0);
   Pulls_3312_4->SetBorderSize(2);
   Pulls_3312_4->SetFrameBorderMode(0);
   
   TH1D *fHistPtPtPullMotherLst_3312_py__4 = new TH1D("fHistPtPtPullMotherLst_3312_py__4","Pt-Pt Pull histo",50,-15,15);

   ci = TColor::GetColor("#000099");
   fHistPtPtPullMotherLst_3312_py__4->SetLineColor(ci);
   fHistPtPtPullMotherLst_3312_py__4->GetXaxis()->SetTitle(" Pt Pull");
   fHistPtPtPullMotherLst_3312_py__4->GetXaxis()->SetLabelFont(42);
   fHistPtPtPullMotherLst_3312_py__4->GetXaxis()->SetTitleFont(42);
   fHistPtPtPullMotherLst_3312_py__4->GetYaxis()->SetLabelFont(42);
   fHistPtPtPullMotherLst_3312_py__4->GetYaxis()->SetTitleFont(42);
   fHistPtPtPullMotherLst_3312_py__4->GetZaxis()->SetLabelFont(42);
   fHistPtPtPullMotherLst_3312_py__4->GetZaxis()->SetTitleOffset(1);
   fHistPtPtPullMotherLst_3312_py__4->GetZaxis()->SetTitleFont(42);
   fHistPtPtPullMotherLst_3312_py__4->Draw("");
   Pulls_3312_4->Modified();
   Pulls_3312->cd();
  
// ------------>Primitives in pad: Pulls_3312_5
   TPad *Pulls_3312_5 = new TPad("Pulls_3312_5", "Pulls_3312_5",0.3433333,0.01,0.6566667,0.49);
   Pulls_3312_5->Draw();
   Pulls_3312_5->cd();
   Pulls_3312_5->Range(0,0,1,1);
   Pulls_3312_5->SetFillColor(0);
   Pulls_3312_5->SetBorderMode(0);
   Pulls_3312_5->SetBorderSize(2);
   Pulls_3312_5->SetFrameBorderMode(0);
   
   TH1D *fHistPtMassPullMotherLst_3312_py__5 = new TH1D("fHistPtMassPullMotherLst_3312_py__5","Pt-Mass Pull histo",50,-5,5);

   ci = TColor::GetColor("#000099");
   fHistPtMassPullMotherLst_3312_py__5->SetLineColor(ci);
   fHistPtMassPullMotherLst_3312_py__5->GetXaxis()->SetTitle(" Mass Pull");
   fHistPtMassPullMotherLst_3312_py__5->GetXaxis()->SetLabelFont(42);
   fHistPtMassPullMotherLst_3312_py__5->GetXaxis()->SetTitleFont(42);
   fHistPtMassPullMotherLst_3312_py__5->GetYaxis()->SetLabelFont(42);
   fHistPtMassPullMotherLst_3312_py__5->GetYaxis()->SetTitleFont(42);
   fHistPtMassPullMotherLst_3312_py__5->GetZaxis()->SetLabelFont(42);
   fHistPtMassPullMotherLst_3312_py__5->GetZaxis()->SetTitleOffset(1);
   fHistPtMassPullMotherLst_3312_py__5->GetZaxis()->SetTitleFont(42);
   fHistPtMassPullMotherLst_3312_py__5->Draw("");
   Pulls_3312_5->Modified();
   Pulls_3312->cd();
  
// ------------>Primitives in pad: Pulls_3312_6
   TPad *Pulls_3312_6 = new TPad("Pulls_3312_6", "Pulls_3312_6",0.6766667,0.01,0.99,0.49);
   Pulls_3312_6->Draw();
   Pulls_3312_6->cd();
   Pulls_3312_6->Range(0,0,1,1);
   Pulls_3312_6->SetFillColor(0);
   Pulls_3312_6->SetBorderMode(0);
   Pulls_3312_6->SetBorderSize(2);
   Pulls_3312_6->SetFrameBorderMode(0);
   
   TH1D *fHistPtEPullMotherLst_3312_py__6 = new TH1D("fHistPtEPullMotherLst_3312_py__6","Pt-E Pull histo",50,-15,15);

   ci = TColor::GetColor("#000099");
   fHistPtEPullMotherLst_3312_py__6->SetLineColor(ci);
   fHistPtEPullMotherLst_3312_py__6->GetXaxis()->SetTitle(" E Pull");
   fHistPtEPullMotherLst_3312_py__6->GetXaxis()->SetLabelFont(42);
   fHistPtEPullMotherLst_3312_py__6->GetXaxis()->SetTitleFont(42);
   fHistPtEPullMotherLst_3312_py__6->GetYaxis()->SetLabelFont(42);
   fHistPtEPullMotherLst_3312_py__6->GetYaxis()->SetTitleFont(42);
   fHistPtEPullMotherLst_3312_py__6->GetZaxis()->SetLabelFont(42);
   fHistPtEPullMotherLst_3312_py__6->GetZaxis()->SetTitleOffset(1);
   fHistPtEPullMotherLst_3312_py__6->GetZaxis()->SetTitleFont(42);
   fHistPtEPullMotherLst_3312_py__6->Draw("");
   Pulls_3312_6->Modified();
   Pulls_3312->cd();
   Pulls_3312->Modified();
   Pulls_3312->cd();
   Pulls_3312->SetSelected(Pulls_3312);
}
