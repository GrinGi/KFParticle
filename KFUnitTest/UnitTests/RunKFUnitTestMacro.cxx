#include "ConfigConstants.h"

#include <TROOT.h>
#include <TSystem.h>
#include <TString.h>
#include <initializer_list>
#include <iostream>

void RunKFUnitTestMacro(const char* installDir = "")
{
#ifndef ALIPHYSICS
    TString prefix = installDir;
    if (prefix.IsNull()) {
        const char* env = gSystem->Getenv("KFPARTICLE_DIR");
        if (env && *env)
            prefix = env;
    }
    // The two test macros below are run from KFUnitTest/UnitTests.
    if (prefix.IsNull())
        prefix = "../..";
    gSystem->ExpandPathName(prefix);

    if (!gSystem->IsAbsoluteFileName(prefix.Data()))
        prefix = TString(gSystem->WorkingDirectory()) + "/" + prefix;

    TString installPrefix;
    TString includeDir;
    TString library;
    for (const TString& base : {prefix, prefix + "/install"}) {
        const TString headers = base + "/include";
        if (gSystem->AccessPathName(headers + "/KFParticle/KFParticle.h") &&
            gSystem->AccessPathName(headers + "/KFParticle.h"))
            continue;

        for (const char* dir : {"lib", "lib64"}) {
            for (const char* ext : {"so", "dylib"}) {
                const TString candidate = base + "/" + dir + "/libKFParticle." + ext;
                if (!gSystem->AccessPathName(candidate.Data())) {
                    library = candidate;
                    break;
                }
            }
            if (!library.IsNull())
                break;
        }
        if (!library.IsNull()) {
            installPrefix = base;
            includeDir = headers;
            break;
        }
    }

    if (library.IsNull()) {
        std::cerr << "Could not find KFParticle headers and library together under "
                  << prefix << " or " << prefix << "/install\n";
        return;
    }

    gSystem->Setenv("KFPARTICLE_PREFIX", installPrefix.Data());
    gSystem->AddIncludePath(TString::Format(" -I\"%s\" -I\"%s/KFParticle\"",
                                            includeDir.Data(), includeDir.Data()).Data());
    if (gSystem->Load(library.Data()) < 0) {
        std::cerr << "Failed to load: " << library << '\n';
        return;
    }
    std::cout << "Loaded " << library << '\n';
    gROOT->ProcessLine("KFParticle start_script_part3324151532;");
#endif

    gSystem->AddIncludePath(" -I/usr/include/eigen3");

    Int_t error = 0;
    gROOT->ProcessLine(".x MakeKFParticleTrees.cxx++", &error);
    if (error != 0) {
        std::cerr << "Error running MakeKFParticleTrees.cxx\n";
        return;
    }

    gROOT->ProcessLine(".x MakeKFUnitTestHistos.cxx++", &error);
    if (error != 0)
        std::cerr << "Error running MakeKFUnitTestHistos.cxx\n";
}
