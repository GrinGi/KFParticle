#pragma once

#include <TMath.h>
#include <TMatrixD.h>
#include <TMatrixDSymEigen.h>
#include <TRandom.h>
#include <TVectorD.h>

#include <Riostream.h>

//#include "../ConfigConstants.h"
#include "KFParticle/KFPVertex.h"
#include "KFParticle/KFParticle.h"

template<typename T>
bool SmearParameters(size_t num_of_params, T parameters[], T covMatArr[])
{

  Int_t ret = 0;

  TMatrixDSym A(num_of_params);
  for (size_t k = 0, i = 0; i < num_of_params; i++) {
    for (size_t j = 0; j <= i; j++, k++) {
      A(i, j) = covMatArr[k];
      A(j, i) = covMatArr[k];
    }
  }

  TMatrixDSymEigen e(A);
  TVectorD D  = e.GetEigenValues();
  TMatrixD V  = e.GetEigenVectors();
  TMatrixD Vt = V;
  Vt.Transpose(V);

  TVectorD X(num_of_params);
  for (size_t i = 0; i < num_of_params; i++) { X(i) = parameters[i]; }

  X           = Vt * X;
  TVectorD X0 = X;
  for (size_t i = 0; i < num_of_params; i++) {
    if (D(i) < 0) { ret = 1; }
    D(i)           = TMath::Abs(D(i));  // correction of the covariance matrix
    Double_t sigma = TMath::Sqrt(D(i));
    X(i) += gRandom->Gaus(0, sigma);
  }
  X  = V * X;
  X0 = V * X0;

  TMatrixD DD(num_of_params, num_of_params);
  for (size_t i = 0; i < num_of_params; i++) {
    for (size_t j = 0; j < num_of_params; j++) { DD(i, j) = 0; }
    DD(i, i) = D(i);
  }
  TMatrixD AA = V * DD * Vt;

  for (size_t k = 0, i = 0; i < num_of_params; i++) {
    parameters[i] = X(i);
    for (size_t j = 0; j <= i; j++, k++) {
      // cout<<AA(i,j) - Cov[k]<<" ";
      covMatArr[k] = AA(i, j);
    }
  }

  return true;
}

bool SmearParticle(KFParticle& part)
{ constexpr int num_of_params = 6;
  constexpr int covmat_size = num_of_params * (num_of_params + 1) / 2;

  float params[num_of_params];
  for (int i = 0; i < num_of_params; i++) { params[i] = part.GetParameter(i); }

  float covmat[covmat_size];
  for (int i = 0; i < covmat_size; i++) { covmat[i] = part.GetCovariance(i); }

  bool res = SmearParameters(num_of_params, params, covmat);

  float mass, masserr;
  part.GetMass(mass, masserr);
  float chi2   = part.GetChi2();
  float charge = (int) part.GetQ();
  int ndf      = part.GetNDF();

#ifdef TESTSUITE
  part.Initialize(params, covmat, charge, chi2, ndf, mass);
#else
  part.Create(params, covmat, charge, mass);
#endif

  return res;
}

bool SmearVertex(KFPVertex& vert)
{ constexpr int num_of_params = 3;
  constexpr int covmat_size = num_of_params * (num_of_params + 1) / 2;

  float params[num_of_params];
  vert.GetXYZ(params);

  float covmat[covmat_size];
  vert.GetCovarianceMatrix(covmat);

  bool res = SmearParameters(num_of_params, params, covmat);
  vert.SetXYZ(params);
  return res;
}
