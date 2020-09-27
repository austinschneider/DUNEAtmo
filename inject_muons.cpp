#include <LeptonInjector/Controller.h>
#include <LeptonInjector/Particle.h>
#include <LeptonInjector/LeptonInjector.h>
#include <LeptonInjector/Constants.h>
#include <earthmodel-service/EarthModelService.h>
#include <string>
#include <memory>

int main(void){

    // define some parameters shared by the injectors
    int n_ranged_events = int(1e6);
    int n_volume_events = int(1e5);

    std::string xs_base = "./csms_differential_v1.0/";

    // specify the final state particles, and construct the first injector
    std::string nu_diff_xs = xs_base + "dsdxdy_nu_CC_iso.fits";
    std::string nu_total_xs = xs_base + "sigma_nu_CC_iso.fits";

    LeptonInjector::Injector numu_ranged_injector(n_ranged_events,
            LeptonInjector::Particle::MuMinus,
            LeptonInjector::Particle::Hadrons,
            nu_diff_xs,
            nu_total_xs,
            true); // is_ranged

    LeptonInjector::Injector numubar_ranged_injector(n_ranged_events,
            LeptonInjector::Particle::MuPlus,
            LeptonInjector::Particle::Hadrons,
            nu_diff_xs,
            nu_total_xs,
            true); // is_ranged

    LeptonInjector::Injector numu_volume_injector(n_volume_events,
            LeptonInjector::Particle::MuMinus,
            LeptonInjector::Particle::Hadrons,
            nu_diff_xs,
            nu_total_xs,
            false); // is_ranged

    LeptonInjector::Injector numubar_volume_injector(n_volume_events,
            LeptonInjector::Particle::MuPlus,
            LeptonInjector::Particle::Hadrons,
            nu_diff_xs,
            nu_total_xs,
            false); // is_ranged

    // some global values shared by all the injectors
    // units come from Constants.h
    double minE = 1e2*LeptonInjector::Constants::GeV;
    double maxE = 1e6*LeptonInjector::Constants::GeV;
    double gamma = 2.;
    double minZenith = 0.*LeptonInjector::Constants::deg;
    double maxZenith = 180.*LeptonInjector::Constants::deg;
    double minAzimuth = 0.*LeptonInjector::Constants::deg;
    double maxAzimuth = 360.*LeptonInjector::Constants::deg;

    // build the Controller object. This will facilitate the simulation itself
    // We need to pass the first injector while building this Controller
    // Dimensions of DUNE module is 58.2 x 3.5 x 12 x 4
    LeptonInjector::Controller cont(numu_ranged_injector,
            minE, maxE,
            gamma,
            minAzimuth, maxAzimuth,
            minZenith, maxZenith,
            30.*LeptonInjector::Constants::m, // injection radius
            30.*LeptonInjector::Constants::m, // injection length
            30.*LeptonInjector::Constants::m, // cylinder radius
            20.*LeptonInjector::Constants::m); // cylinder height


    cont.AddInjector(numubar_ranged_injector);
    cont.AddInjector(numu_volume_injector);
    cont.AddInjector(numubar_volume_injector);

    std::string path;
    if (const char* env_p = getenv("GOLEMSOURCEPATH")){
        path = std::string( env_p ) + "/LeptonInjectorDUNE/";
    }
    else {
        path = "./";
    }
    path += "resources/earthparams/";
    std::cout << path << std::endl;

    earthmodel::EarthModelService earthModel(
        "DUNE",
        path,
        std::vector<std::string>({"PREM_dune"}),
        std::vector<std::string>({"Standard"}),
        "NoIce",
        20.0*LeptonInjector::Constants::degrees,
        1480.0*LeptonInjector::Constants::m);

    cont.SetEarthModel(std::shared_ptr<earthmodel::EarthModelService>(&earthModel));

    cont.NameOutfile("./data_output_DUNE.h5");
    cont.NameLicFile("./config_DUNE.lic");

    // Run the program.
    cont.Execute();
}
