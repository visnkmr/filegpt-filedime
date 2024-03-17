{ pkgs ? import <nixpkgs> { config.allowUnfree = true; config.cudaSupport = true; } }:

pkgs.mkShell rec{

  nativeBuildInputs = with pkgs; [
    python311
    stdenv.cc.cc.lib
    libstdcxx5
    cudatoolkit
    linuxPackages.nvidia_x11
  ];
  

  buildInputs = [ pkgs.docker];

  venvDir = "filedimegpt";

  shellHook = ''
  
    set -h #remove "bash: hash: hashing disabled" warning !
    SOURCE_DATE_EPOCH=$(date +%s)
    if ! [ -d "${venvDir}" ]; then
      python -m venv "${venvDir}"
    fi
    source "${venvDir}/bin/activate"
       export CUDA_PATH=${pkgs.cudatoolkit}
    export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.ncurses5}/lib:${pkgs.stdenv.cc.cc.lib}/lib # WSL CASE
    # export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
    export EXTRA_CCFLAGS="-I/usr/include"
      echo ================================================
      echo Nix devenv shell for Network Intrusion Detection
      echo To add python modules use 'poetry add'
      echo ================================================

  '';
}