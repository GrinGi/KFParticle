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

#ifndef KFParticleDef_H
#define KFParticleDef_H

#include <cstring>
#include <vector>

#if defined(HLTCA_STANDALONE) || defined(CBM_ONLINE)
#include "RootTypesDef.h"
#else
#include "TObject.h"
#endif

#include "KFPSimd.h"
#include "KFPSimdPaddedVector.h"

#define NInputSets 8

namespace kfp
{
  using ::kfp::simd::float_m;
  using ::kfp::simd::float_v;
  using ::kfp::simd::int_m;
  using ::kfp::simd::int_v;
  using ::kfp::simd::PaddedVector;
  using ::kfp::simd::select;
  using ::kfp::simd::SimdAlignmentF;
  using ::kfp::simd::SimdAlignmentI;
  using ::kfp::simd::SimdLen;
}  // namespace kfp

#if defined(HLTCA_STANDALONE)
typedef unsigned char UChar_t;
typedef UChar_t Byte_t;
typedef int Int_t;
typedef double Double_t;
#elif defined(CBM_ONLINE)
#else
#include "Rtypes.h"
#endif

#endif
