# Release and archive checklist

This checklist connects the GitHub release, the Zenodo archive, and the
version cited by the manuscript.

## Before the first release

- [ ] Verify every checksum from the completed local tree.
- [ ] Initialize Git and commit the complete tree.
- [ ] Clone the repository into a fresh directory.
- [ ] In the fresh clone, run `sha256sum -c SHA256SUMS`.
- [ ] In the fresh clone, run `python3 phase_count_audit.py`.
- [ ] In the fresh clone, run `python3 numerics/reproduce_numerics.py`.
- [ ] In the fresh clone, run `python3 length_sweeps.py --calibrate`.
- [ ] Confirm that the copied scientific payload is byte-identical to the
      verified v0.4.2 manuscript package.

## GitHub and Zenodo order

- [ ] Create the public GitHub repository
      `JeromeNicolas2026/signed-delay-nodal-pumpkin-chains`.
- [ ] Push the verified commit to the `main` branch.
- [ ] Clone the public GitHub repository and repeat the checksum check.
- [ ] Connect Zenodo to GitHub and enable this repository **before** creating
      the first GitHub release.
- [ ] Create the immutable GitHub release and tag `v1.0.0`.
- [ ] Wait for Zenodo to ingest that release.
- [ ] Record the DOI for the specific `v1.0.0` record, not the concept DOI.

If an error is found after Zenodo has minted the DOI, do not delete and
recreate `v1.0.0`. Correct the repository and publish `v1.0.1`.

## Verify the object that will be cited

- [ ] Download the archive from the Zenodo record itself.
- [ ] Extract it into a new directory.
- [ ] Run `sha256sum -c SHA256SUMS` in the extracted archive.
- [ ] Run `python3 phase_count_audit.py`.
- [ ] Run `python3 numerics/reproduce_numerics.py`.
- [ ] Confirm that the archive produces the two exact counting values and the
      exact one-tailed correlation printed in `README.md`.

## Link the archive to the manuscript

- [ ] Add the version DOI to the manuscript's Data Availability statement.
- [ ] Add the version-specific software/data citation to the bibliography.
- [ ] Resolve the DOI in a browser and verify that it opens the `v1.0.0`
      Zenodo record.
- [ ] Confirm that every numerical value cited in the manuscript is produced
      by that downloaded Zenodo archive.
- [ ] Recompile the manuscript and check references, citations, and layout.

After the article is accepted, add its journal link to `README.md` in a new
repository revision. Do not alter the already archived `v1.0.0` record.
