/*
 * This file is part of KFParticle package
 * Copyright (C) 2007-2019 FIAS Frankfurt Institute for Advanced Studies
 *               2007-2019 Goethe University of Frankfurt
 *               2007-2019 Ivan Kisel <I.Kisel@compeng.uni-frankfurt.de>
 *               2007-2019 Maksym Zyzak
 *
 * KFParticle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * KFParticle is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef KFPTrackVector_H
#define KFPTrackVector_H

#include "KFPTrack.h"
#include "KFParticleDef.h"

#include <cstring>

/** @class KFPTrackVector
 ** @brief A class to store vectors of input tracks in the cartesian
 *parametrisation.
 ** @author  M.Zyzak, I.Kisel
 ** @date 05.02.2019
 ** @version 1.0
 **
 ** A track is described with the state vector { X, Y, Z, Px, Py, Pz }
 ** and the corresponding covariance matrix. Also contains charge of the track,
 ** unique Id, assigned PDG hypothesis, charge, index of the corresponding
 ** primary vertex in case of primary track, number of hits from precise
 ** detectors (like MVD in CBM, HFT in STAR, ITS in ALICE, etc.)
 ** and the magnetic field approximation along the track trajectory
 ** (in case of nonhomogeneous CBM-like field). \n
 ** The data model implemented in the class is "Structure Of Arrays":
 ** each parameter is stroed in a separate vector. Such data structure
 ** allows fast vectorised access to the aligned data providing the
 ** maximum possible speed for data reading, and at the same time easy
 ** random access to the data members. Tracks are sorted by
 *KFParticleTopoReconstructor::SortTracks():
 ** electrons, muons, pions, tracks without PID, kaons, protons, deuterons,
 *tritons, He3, He4, He6, Li6, Li7, Be7, Sigmas.
 **/

class KFPTrackVector {
  friend class KFParticleTopoReconstructor;

 public:
  using float_v = ::kfp::float_v;
  using int_v   = ::kfp::int_v;
  template<typename T>
  using PaddedVector = ::kfp::simd::PaddedVector<T>;

  static constexpr auto SimdLen = ::kfp::SimdLen;

  KFPTrackVector()
    : fId()
    , fPDG()
    , fQ()
    , fPVIndex()
    , fNPixelHits()
    , fNE(0)
    , fNMu(0)
    , fNPi(0)
    , fNK(0)
    , fNP(0)
    , fND(0)
    , fNT(0)
    , fNHe3(0)
    , fNHe4(0)
    , fNHe6(0)
    , fNLi6(0)
    , fNLi7(0)
    , fNBe7(0)
    , fNSigma(0)
  {
  }
  ~KFPTrackVector() {}

  /**Returns size of the vectors. All data vectors have the same size. */
  int Size() const { return fP[0].size(); }
  int DataSize() const
  {
    /**Returns size of the memory in floats (4 bytes or 32 bits) allocated by
     * the current object. */
    const int& size = fP[0].size();

    const int dataSize = size * 32
#ifdef NonhomogeneousField
                         + size * 10
#endif
                         + 9;
    return dataSize;
  }

  void Resize(const int n);
  void Set(KFPTrackVector& v, int vSize, int offset);
  void SetTracks(const KFPTrackVector& track, const PaddedVector<int_v>& trackIndex, const int nIndexes);
  void GetTrack(KFPTrack& track, const int n);

  const PaddedVector<float_v>& X() const
  { return fP[0]; }  ///< Returns constant reference to the vector with X coordinates.
  const PaddedVector<float_v>& Y() const
  { return fP[1]; }  ///< Returns constant reference to the vector with Y coordinates.
  const PaddedVector<float_v>& Z() const
  { return fP[2]; }  ///< Returns constant reference to the vector with Z coordinates.
  const PaddedVector<float_v>& Px() const
  { return fP[3]; }  ///< Returns constant reference to the vector with Px components of
  ///< momentum.
  const PaddedVector<float_v>& Py() const
  { return fP[4]; }  ///< Returns constant reference to the vector with Py components of
  ///< momentum.
  const PaddedVector<float_v>& Pz() const
  { return fP[5]; }  ///< Returns constant reference to the vector with Pz components of
  ///< momentum.

  const PaddedVector<float_v>& Parameter(const int i) const
  { return fP[i]; }  ///< Returns constant reference to the track parameter vector with index
  ///< "i".
  const PaddedVector<float_v>& Covariance(const int i) const
  { return fC[i]; }  ///< Returns constant reference to the vector of the covariance matrix
  ///< elements with index "i".
#ifdef NonhomogeneousField
  const PaddedVector<float_v>& FieldCoefficient(const int i) const
  { return fField[i]; }  ///< Returns constant reference to the magnetic field coefficient with index
  ///< "i".
#endif

  const PaddedVector<int_v>& Id() const { return fId; }  ///< Returns constant reference to the vector with track Id
  ///< KFPTrackVector::fId.
  const PaddedVector<int_v>& PDG() const
  { return fPDG; }  ///< Returns constant reference to the vector with assigned PDG hypothesis
  ///< KFPTrackVector::fPDG.
  const PaddedVector<int_v>& Q() const { return fQ; }  ///< Returns constant reference to the vector with charge
  ///< KFPTrackVector::fQ.
  const PaddedVector<int_v>& PVIndex() const
  { return fPVIndex; }  ///< Returns constant reference to the vector with indices of corresponding
  ///< primary vertex KFPTrackVector::fPVIndex.
  const PaddedVector<int_v>& NPixelHits() const
  { return fNPixelHits; }  ///< Returns constant reference to the vector with the number of precise
  ///< measurements KFPTrackVector::fNPixelHits.

  float Pt(const int n) const
  {
    return sqrt(fP[3][n] * fP[3][n] + fP[4][n] * fP[4][n]);
  }  ///< Returns transverse momentum of the track with index "n".
  float P(const int n) const
  {
    return sqrt(fP[3][n] * fP[3][n] + fP[4][n] * fP[4][n] + fP[5][n] * fP[5][n]);
  }  ///< Returns momentum of the track with index "n".

  // modifiers
  void SetParameter(float value, int iP, int iTr)
  { fP[iP][iTr] = value; }  ///< Sets the "value" of the parameter "iP" of the track with index "iTr".
  void SetCovariance(float value, int iC, int iTr)
  { fC[iC][iTr] = value; }  ///< Sets the "value" of the element of covariance matrix "iC" of the track
  ///< with index "iTr".

  void SetParameter(const float_v& value, int iP, int iTr);
  void SetCovariance(const float_v& value, int iC, int iTr);

#ifdef NonhomogeneousField
  void SetFieldCoefficient(float value, int iP, int iTr)
  { fField[iP][iTr] = value; }  ///< Sets the "value" of the field coefficient "iP" of the track with index
  ///< "iTr".
#endif
  void SetId(int value, int iTr) { fId[iTr] = value; }    ///< Sets Id of the track with index "iTr".
  void SetPDG(int value, int iTr) { fPDG[iTr] = value; }  ///< Sets PDG hypothesis of the track with index "iTr".
  void SetQ(int value, int iTr) { fQ[iTr] = value; }      ///< Sets charge of the track with index "iTr".
  void SetPVIndex(int value, int iTr)
  { fPVIndex[iTr] = value; }  ///< Sets index of the corresponding primary vertex of the track with index
  ///< "iTr".
  void SetNPixelHits(int value, int iTr)
  { fNPixelHits[iTr] = value; }              ///< Sets number of precise measurement of the track with index "iTr".
  void SetLastElectron(int n) { fNE = n; }   ///< Sets index of the last electron.
  void SetLastMuon(int n) { fNMu = n; }      ///< Sets index of the last muon.
  void SetLastPion(int n) { fNPi = n; }      ///< Sets index of the last pion.
  void SetLastKaon(int n) { fNK = n; }       ///< Sets index of the last kaon.
  void SetLastProton(int n) { fNP = n; }     ///< Sets index of the last proton.
  void SetLastDeuteron(int n) { fND = n; }   ///< Sets index of the last deuteron.
  void SetLastTritium(int n) { fNT = n; }    ///< Sets index of the last triton.
  void SetLastHe3(int n) { fNHe3 = n; }      ///< Sets index of the last He3.
  void SetLastHe4(int n) { fNHe4 = n; }      ///< Sets index of the last He4.
  void SetLastHe6(int n) { fNHe6 = n; }      ///< Sets index of the last He6.
  void SetLastLi6(int n) { fNLi6 = n; }      ///< Sets index of the last Li6.
  void SetLastLi7(int n) { fNLi7 = n; }      ///< Sets index of the last Li7.
  void SetLastBe7(int n) { fNBe7 = n; }      ///< Sets index of the last Be7.
  void SetLastSigma(int n) { fNSigma = n; }  ///< Sets index of the last Sigma.

  void RecalculateLastIndex()
  {
    /** Recalculate the last index of each track specie. Should be called after
     * track sorting. */
    fNE     = 0;
    fNMu    = 0;
    fNPi    = 0;
    fNK     = 0;
    fNP     = 0;
    fND     = 0;
    fNT     = 0;
    fNHe3   = 0;
    fNHe4   = 0;
    fNHe6   = 0;
    fNLi6   = 0;
    fNLi7   = 0;
    fNBe7   = 0;
    fNSigma = 0;
    for (int i = 0; i < Size(); i++) {
      switch (abs(fPDG[i])) {
        case 11: fNE++; break;
        case 13: fNMu++; break;
        case 19: fNMu++; break;
        case 211: fNPi++; break;
        case 1: fNPi++; break;
        case 3: fNPi++; break;
        case 321: fNK++; break;
        case 2212: fNP++; break;
        case 1000010020: fND++; break;
        case 1000010030: fNT++; break;
        case 1000020030: fNHe3++; break;
        case 1000020040: fNHe4++; break;
        case 1000020060: fNHe6++; break;
        case 1000030060: fNLi6++; break;
        case 1000030070: fNLi7++; break;
        case 1000040070: fNBe7++; break;
        case 2000003112: fNSigma++; break;
      }
    }

    fNMu += fNE;
    fNPi += fNMu;
    fNK += fNPi;
    fNP += fNK;
    fND += fNP;
    fNT += fND;
    fNHe3 += fNT;
    fNHe4 += fNHe3;
    fNHe6 += fNHe4;
    fNLi6 += fNHe6;
    fNLi7 += fNLi6;
    fNBe7 += fNLi7;
    fNSigma += fNBe7;
  }

  int FirstElectron() { return 0; }                ///< Returns index of the first electron.
  const int& LastElectron() const { return fNE; }  ///< Returns index of the last electron.
  int NElectrons() { return fNE; }                 ///< Returns number of electrons.
  int FirstMuon()
  { return int(fNE / SimdLen) * SimdLen; }  ///< Returns index of the first element of the SIMD vector with the first
  ///< muon.
  const int& LastMuon() const { return fNMu; }  ///< Returns index of the last muon.
  int NMuons() { return fNMu - fNE; }           ///< Returns number of muons.
  int FirstPion()
  { return int(fNMu / SimdLen) * SimdLen; }  ///< Returns index of the first element of the SIMD vector with the first
  ///< pion.
  const int& LastPion() const { return fNPi; }  ///< Returns index of the last pion.
  int NPions() { return fNPi - fNMu; }          ///< Returns number of pions.
  int FirstKaon()
  { return int(fNPi / SimdLen) * SimdLen; }  ///< Returns index of the first element of the SIMD vector with the first
  ///< kaon.
  const int& LastKaon() const { return fNK; }  ///< Returns index of the last kaon.
  int NKaons() { return fNK - fNPi; }          ///< Returns number of kaons.
  int FirstProton()
  { return int(fNK / SimdLen) * SimdLen; }  ///< Returns index of the first element of the SIMD vector with the first
  ///< proton.
  const int& LastProton() const { return fNP; }  ///< Returns index of the last proton.
  int NProtons() { return fNP - fNK; }           ///< Returns number of protons.
  int FirstDeuteron()
  { return int(fNP / SimdLen) * SimdLen; }  ///< Returns index of the first element of the SIMD vector with the first
  ///< deuteron.
  const int& LastDeuteron() const { return fND; }  ///< Returns index of the last deuteron.
  int NDeuterons() { return fND - fNP; }           ///< Returns number of deuterons.
  int FirstTritium()
  { return int(fND / SimdLen) * SimdLen; }  ///< Returns index of the first element of the SIMD vector with the first
  ///< triton.
  const int& LastTritium() const { return fNT; }  ///< Returns index of the last triton.
  int NTritiums() { return fNT - fND; }           ///< Returns number of tritons.
  int FirstHe3()
  { return int(fNT / SimdLen) * SimdLen; }  ///< Returns index of the first element of the SIMD vector with the first
  ///< He3.
  const int& LastHe3() const { return fNHe3; }  ///< Returns index of the last He3.
  int NHe3s() { return fNHe3 - fNT; }           ///< Returns number of He3 tracks.
  int FirstHe4()
  { return int(fNHe3 / SimdLen) * SimdLen; }  ///< Returns index of the first element of the SIMD vector with the first
  ///< He4.
  const int& LastHe4() const { return fNHe4; }  ///< Returns index of the last He4.
  int NHe4s() { return fNHe4 - fNHe3; }         ///< Returns number of He4 tracks.
  int FirstHe6()
  { return int(fNHe4 / SimdLen) * SimdLen; }  ///< Returns index of the first element of the SIMD vector with the first
  ///< He6.
  const int& LastHe6() const { return fNHe6; }  ///< Returns index of the last He6.
  int NHe6s() { return fNHe6 - fNHe4; }         ///< Returns number of He6 tracks.
  int FirstLi6()
  { return int(fNHe6 / SimdLen) * SimdLen; }  ///< Returns index of the first element of the SIMD vector with the first
  ///< Li6.
  const int& LastLi6() const { return fNLi6; }  ///< Returns index of the last Li6.
  int NLi6s() { return fNLi6 - fNHe6; }         ///< Returns number of Li6 tracks.
  int FirstLi7()
  { return int(fNLi6 / SimdLen) * SimdLen; }  ///< Returns index of the first element of the SIMD vector with the first
  ///< Li7.
  const int& LastLi7() const { return fNLi7; }  ///< Returns index of the last Li7.
  int NLi7s() { return fNLi7 - fNLi6; }         ///< Returns number of Li7 tracks.
  int FirstBe7()
  { return int(fNLi7 / SimdLen) * SimdLen; }  ///< Returns index of the first element of the SIMD vector with the first
  ///< Be7.
  const int& LastBe7() const { return fNBe7; }  ///< Returns index of the last Be7.
  int NBe7s() { return fNBe7 - fNLi7; }         ///< Returns number of Be7 tracks.
  int FirstSigma()
  { return int(fNBe7 / SimdLen) * SimdLen; }  ///< Returns index of the first element of the SIMD vector with the first
  ///< Sigma.
  const int& LastSigma() const { return fNSigma; }  ///< Returns index of the last Sigma.
  int NSigmas() { return fNSigma - fNBe7; }         ///< Returns number of Sigma tracks.

  void AddElectron() { fNE++; }   ///< Increases by one index of the last electron.
  void AddMuon() { fNMu++; }      ///< Increases by one index of the last muon.
  void AddPion() { fNPi++; }      ///< Increases by one index of the last pion.
  void AddKaon() { fNK++; }       ///< Increases by one index of the last kaon.
  void AddProton() { fNP++; }     ///< Increases by one index of the last proton.
  void AddDeuteron() { fND++; }   ///< Increases by one index of the last deuteron.
  void AddTririum() { fNT++; }    ///< Increases by one index of the last triton.
  void AddHe3() { fNHe3++; }      ///< Increases by one index of the last He3.
  void AddHe4() { fNHe4++; }      ///< Increases by one index of the last He4.
  void AddHe6() { fNHe6++; }      ///< Increases by one index of the last He6.
  void AddLi6() { fNLi6++; }      ///< Increases by one index of the last Li6.
  void AddLi7() { fNLi7++; }      ///< Increases by one index of the last Li7.
  void AddBe7() { fNBe7++; }      ///< Increases by one index of the last Be7.
  void AddSigma() { fNSigma++; }  ///< Increases by one index of the last Sigma.

  void RotateXY(float_v alpha, int firstElement);

  void PrintTrack(int n);
  void Print();

  const KFPTrackVector& operator=(const KFPTrackVector& track)
  {
    /** Operator to copy one KFPTrackVector object to another. Makes one-to-one
     * copy.*/
    const int localSize = track.Size();

    for (int i = 0; i < 6; i++) {
      fP[i].resize(localSize);
      for (int n = 0; n < localSize; n++) { fP[i][n] = track.fP[i][n]; }
    }

    for (int i = 0; i < 21; i++) {
      fC[i].resize(localSize);
      for (int n = 0; n < localSize; n++) { fC[i][n] = track.fC[i][n]; }
    }

#ifdef NonhomogeneousField
    for (int i = 0; i < 10; i++) {
      fField[i].resize(localSize);
      for (int n = 0; n < localSize; n++) { fField[i][n] = track.fField[i][n]; }
    }
#endif

    fId.resize(localSize);
    for (int n = 0; n < localSize; n++) { fId[n] = track.fId[n]; }

    fPDG.resize(localSize);
    for (int n = 0; n < localSize; n++) { fPDG[n] = track.fPDG[n]; }

    fQ.resize(localSize);
    for (int n = 0; n < localSize; n++) { fQ[n] = track.fQ[n]; }

    fPVIndex.resize(localSize);
    for (int n = 0; n < localSize; n++) { fPVIndex[n] = track.fPVIndex[n]; }

    fNPixelHits.resize(localSize);
    for (int n = 0; n < localSize; n++) { fNPixelHits[n] = track.fNPixelHits[n]; }

    fNE     = track.fNE;
    fNMu    = track.fNMu;
    fNPi    = track.fNPi;
    fNK     = track.fNK;
    fNP     = track.fNP;
    fND     = track.fND;
    fNT     = track.fNT;
    fNHe3   = track.fNHe3;
    fNHe4   = track.fNHe4;
    fNHe6   = track.fNHe6;
    fNLi6   = track.fNLi6;
    fNLi7   = track.fNLi7;
    fNBe7   = track.fNBe7;
    fNSigma = track.fNSigma;

    return *this;
  }


  void SetDataToBuffer(int* idata, int& offset) const
  {
    /** Copies entire vector to the provided memory starting form the position
     *"offset".
     ** The function is used in KFPInputData::SetDataToVector().
     ** \param[out] data - pointer to the memory where the track vectors should
     *be copied; since all fields of
     ** KFPTrackVector are of the same size (int or float) pointer can be safely
     *casted to int*
     ** \param[in,out] offset - starting position in "data" where vectors should
     *be copied; after all vectors are
     ** copied the offset is shifted on the size of the written object so the
     *next KFPTrackVector object
     ** can be copied to the "data"
     **/

    const int size = Size();
    for (int iP = 0; iP < 6; iP++) {
      for (int n = 0; n < size; n++) {
        const float value = fP[iP][n];
        std::memcpy(&idata[offset + n], &value, sizeof(value));
      }
      offset += size;
    }

    for (int iC = 0; iC < 21; iC++) {
      for (int n = 0; n < size; n++) {
        const float value = fC[iC][n];
        std::memcpy(&idata[offset + n], &value, sizeof(value));
      }
      offset += size;
    }

    for (int n = 0; n < size; n++) { idata[offset + n] = fId[n]; }
    offset += size;

    for (int n = 0; n < size; n++) { idata[offset + n] = fPDG[n]; }
    offset += size;

    for (int n = 0; n < size; n++) { idata[offset + n] = fQ[n]; }
    offset += size;

    for (int n = 0; n < size; n++) { idata[offset + n] = fPVIndex[n]; }
    offset += size;

    for (int n = 0; n < size; n++) { idata[offset + n] = fNPixelHits[n]; }
    offset += size;

#ifdef NonhomogeneousField
    for (int iF = 0; iF < 10; iF++) {
      for (int n = 0; n < size; n++) {
        const float value = fField[iF][n];
        std::memcpy(&idata[offset + n], &value, sizeof(value));
      }
      offset += size;
    }
#endif

    idata[offset++] = fNE;
    idata[offset++] = fNMu;
    idata[offset++] = fNPi;
    idata[offset++] = fNK;
    idata[offset++] = fNP;
    idata[offset++] = fND;
    idata[offset++] = fNT;
    idata[offset++] = fNHe3;
    idata[offset++] = fNHe4;
    idata[offset++] = fNHe6;
    idata[offset++] = fNLi6;
    idata[offset++] = fNLi7;
    idata[offset++] = fNBe7;
    idata[offset++] = fNSigma;
  }

  void ReadDataFromBuffer(const int* idata, int& offset)
  {
    /** Copies entire vector from the provided memory starting form the position
     *"offset".
     ** The function is used in KFPInputData::ReadDataFromVector().
     ** \param[in] data - pointer to the memory with the track vectors; since
     *all fields of
     ** KFPTrackVector are of the same size (int or float) pointer can be safely
     *casted to int*
     ** \param[in,out] offset - starting position of the memory to be copied;
     *after all vectors are
     ** copied the offset is shifted on the size of the read object so the next
     *KFPTrackVector object
     ** can be copied
     **/

    const int size = Size();

    for (int iP = 0; iP < 6; iP++) {
      for (int n = 0; n < size; n++) {
        float value;
        std::memcpy(&value, &idata[offset + n], sizeof(value));
        fP[iP][n] = value;
      }
      fP[iP].fillPaddingWithLastElement();
      offset += size;
    }

    for (int iC = 0; iC < 21; iC++) {
      for (int n = 0; n < size; n++) {
        float value;
        std::memcpy(&value, &idata[offset + n], sizeof(value));
        fC[iC][n] = value;
      }
      fC[iC].fillPaddingWithLastElement();
      offset += size;
    }

    for (int n = 0; n < size; n++) { fId[n] = idata[offset + n]; }
    fId.fillPaddingWithLastElement();
    offset += size;

    for (int n = 0; n < size; n++) { fPDG[n] = idata[offset + n]; }
    fPDG.fillPaddingWithLastElement();
    offset += size;

    for (int n = 0; n < size; n++) { fQ[n] = idata[offset + n]; }
    fQ.fillPaddingWithLastElement();
    offset += size;

    for (int n = 0; n < size; n++) { fPVIndex[n] = idata[offset + n]; }
    fPVIndex.fillPaddingWithLastElement();
    offset += size;

    for (int n = 0; n < size; n++) { fNPixelHits[n] = idata[offset + n]; }
    fNPixelHits.fillPaddingWithLastElement();
    offset += size;

#ifdef NonhomogeneousField
    for (int iF = 0; iF < 10; iF++) {
      for (int n = 0; n < size; n++) {
        float value;
        std::memcpy(&value, &idata[offset + n], sizeof(value));
        fField[iF][n] = value;
      }
      fField[iF].fillPaddingWithLastElement();
      offset += size;
    }
#endif

    fNE     = idata[offset++];
    fNMu    = idata[offset++];
    fNPi    = idata[offset++];
    fNK     = idata[offset++];
    fNP     = idata[offset++];
    fND     = idata[offset++];
    fNT     = idata[offset++];
    fNHe3   = idata[offset++];
    fNHe4   = idata[offset++];
    fNHe6   = idata[offset++];
    fNLi6   = idata[offset++];
    fNLi7   = idata[offset++];
    fNBe7   = idata[offset++];
    fNSigma = idata[offset++];
  }

 private:
  PaddedVector<float_v> fP[6];   ///< Vectors with parameters of the track : X, Y, Z, Px, Py, Pz.
  PaddedVector<float_v> fC[21];  ///< Vectors with covariance matrix of the track
                                 ///< parameters stroed in a lower triangular form.

  PaddedVector<int_v> fId;          ///< Vector with the unique Id of tracks.
  PaddedVector<int_v> fPDG;         ///< Vector with the PDG hypothesis.
  PaddedVector<int_v> fQ;           ///< Vector with the charge of the tracks.
  PaddedVector<int_v> fPVIndex;     ///< Vector with the index of the corresponding primary vertex.
                                    ///< If track is considered secondary "-1" is stored.
  PaddedVector<int_v> fNPixelHits;  ///< Vector with the number of hits from precise detectors
                                    ///< (like MVD in CBM, HFT in STAR, ITS in ALICE, etc.)

  /** The coefficients of the field approximation of each field component along
   *the track trajectory using parabolas: \n
   ** cx0 = fField[0], cx1 = fField[1], cx2 = fField[2] - coefficients of the Bx
   *approximation; \n
   ** cy0 = fField[3], cy1 = fField[4], cy2 = fField[5] - coefficients of the By
   *approximation; \n
   ** cz0 = fField[6], cz1 = fField[7], cz2 = fField[8] - coefficients of the Bz
   *approximation; \n
   ** z0 = fField[9] - reference Z coordinate. \n
   ** Bx(z) = cx0 + cx1*(z-z0) + cx2*(z-z0)^2 \n
   ** By(z) = cy0 + cy1*(z-z0) + cy2*(z-z0)^2 \n
   ** Bz(z) = cz0 + cz1*(z-z0) + cz2*(z-z0)^2
   **/
#ifdef NonhomogeneousField
  PaddedVector<float_v> fField[10];
#endif

  int fNE;      ///< Index of the last electron.
  int fNMu;     ///< Index of the last muon.
  int fNPi;     ///< Index of the last pion (plus tracks without PID).
  int fNK;      ///< Index of the last kaon.
  int fNP;      ///< Index of the last proton.
  int fND;      ///< Index of the last deuteron.
  int fNT;      ///< Index of the last triton.
  int fNHe3;    ///< Index of the last He3.
  int fNHe4;    ///< Index of the last He4.
  int fNHe6;    ///< Index of the last He6.
  int fNLi6;    ///< Index of the last Li6.
  int fNLi7;    ///< Index of the last Li7.
  int fNBe7;    ///< Index of the last Be7.
  int fNSigma;  ///< Index of the last Sigma.
};

inline void KFPTrackVector::SetParameter(const float_v& value, int iP, int iTr)
{
  /** Copies the SIMD vector "value" to the parameter vector
   *KFPTrackVector::fP[iP]
   ** starting at the position "iTr".
   ** \param[in] value - SIMD vector with the values to be stored
   ** \param[in] iP - number of the parameter vector
   ** \param[in] iTr - starting position in the parameter vector where the
   *values should be stored
   **/

  assert(iTr >= 0 && iTr < Size() && iTr % SimdLen == 0);

  fP[iP].block(iTr / SimdLen) = value;
}

inline void KFPTrackVector::SetCovariance(const float_v& value, int iC, int iTr)
{
  /** Copies the SIMD vector "value" to the element of the covariance matrix
   *vector KFPTrackVector::fC[iC]
   ** starting at the position "iTr".
   ** \param[in] value - SIMD vector with the values to be stored
   ** \param[in] iC - number of the element of the covariance matrix
   ** \param[in] iTr - starting position in the parameter vector where the
   *values should be stored
   **/

  assert(iTr >= 0 && iTr < Size() && iTr % SimdLen == 0);
  fC[iC].block(iTr / SimdLen) = value;
}

inline void KFPTrackVector::GetTrack(KFPTrack& track, const int n)
{
  /** Copies track with index "n" for the current object to the KFPTrack object
   *"track".
   ** \param[out] track - KFPTrack object, where track with index "n" is copied
   ** \param[in] n - index of the track to be copied
   **/
  track.SetParameters(fP[0][n], fP[1][n], fP[2][n], fP[3][n], fP[4][n], fP[5][n]);

  track.SetCovariance(0, fC[0][n]);
  track.SetCovariance(1, fC[1][n]);
  track.SetCovariance(2, fC[2][n]);
  track.SetCovariance(3, fC[3][n]);
  track.SetCovariance(4, fC[4][n]);
  track.SetCovariance(5, fC[5][n]);
  track.SetCovariance(6, fC[6][n]);
  track.SetCovariance(7, fC[7][n]);
  track.SetCovariance(8, fC[8][n]);
  track.SetCovariance(9, fC[9][n]);
  track.SetCovariance(10, fC[10][n]);
  track.SetCovariance(11, fC[11][n]);
  track.SetCovariance(12, fC[12][n]);
  track.SetCovariance(13, fC[13][n]);
  track.SetCovariance(14, fC[14][n]);
  track.SetCovariance(15, fC[15][n]);
  track.SetCovariance(16, fC[16][n]);
  track.SetCovariance(17, fC[17][n]);
  track.SetCovariance(18, fC[18][n]);
  track.SetCovariance(19, fC[19][n]);
  track.SetCovariance(20, fC[20][n]);

  //     track.SetChi2(fChi2[n]);
  //     track.SetNDF(fNDF[n]);
  track.SetId(fId[n]);
  track.SetCharge(fQ[n]);

#ifdef NonhomogeneousField
  for (int i = 0; i < 10; i++) { track.SetFieldCoeff(fField[i][n], i); }
#endif
}
#endif
