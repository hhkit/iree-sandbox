{
  description = "A devShell example";

  inputs = {
    nixpkgs.url      = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url  = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python314;
        pythonPackages = python.pkgs;
        lib-path = with pkgs; lib.makeLibraryPath [
          libffi
          openssl
          stdenv.cc.cc
        ];
      in with pkgs; 
      {
        devShell = mkShell rec {
          packages = [
          ];

          buildInputs = [
            readline
            libffi
            openssl
            git
            openssh
            rsync
          ];

          shellHook = ''
            SOURCE_DATE_EPOCH=$(date +%s)
            export "LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${lib-path}"
            VENV=.venv

            if test ! -d $VENV; then
              python3 -m venv $VENV
            fi
            source ./$VENV/bin/activate
            export PYTHONPATH=`pwd`/$VENV/${python.sitePackages}/:$PYTHONPATH
            pip install -r requirements.txt
          '';

          postShellHook = ''
            ln -sf ${python.sitePackages}/* ./.venv/lib/python3.14/site-packages
          '';
        };
      }
    );
  }