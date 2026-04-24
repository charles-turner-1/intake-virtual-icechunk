from pathlib import Path

import intake
import pytest
import virtualizarr
from intake_esm import esm_datastore
from pandas.testing import assert_frame_equal

from intake_virtual_icechunk.source import IcechunkStoreBuilder

__all__ = ["IcechunkStoreBuilder", "pytest"]


class TestIcechunkStoreBuilder:
    """
    Tests for the IcechunkStoreBuilder class, which is responsible for building
    an IcechunkStore from a given intake-esm datastore.
    """

    @pytest.fixture
    def intake_esm_kwargs(self) -> dict[str, list[str]]:
        return {
            "columns_with_iterables": [
                "variable",
                "variable_long_name",
                "variable_standard_name",
                "variable_cell_methods",
                "variable_units",
            ]
        }

    @pytest.fixture
    def om2_esmds_path(self, sample_data: Path) -> Path:
        """
        Fixture that provides the path to the intake-esm datastore for the OM2 model.
        """
        return sample_data / "access-om2" / "esmcat" / "access-om2.json"

    @pytest.fixture
    def om2_datastore(self, om2_esmds_path, intake_esm_kwargs) -> esm_datastore:
        """
        Fixture that provides an intake-esm datastore for the OM2 model.
        """

        return intake.open_esm_datastore(
            str(om2_esmds_path),
            **intake_esm_kwargs,
        )

    def test_init_infer_parser(
        self, om2_esmds_path, om2_datastore, intake_esm_kwargs, tmpdir
    ):
        """
        Initialisation without a parser should trigger parser inference, which
        in turn should open the esm datastore
        """
        dummy_store_path = tmpdir / "dummy_store.icechunk"
        builder = IcechunkStoreBuilder(
            om2_esmds_path, intake_esm_kwargs, dummy_store_path
        )

        assert isinstance(builder.parser, virtualizarr.parsers.hdf.hdf.HDFParser)
        assert_frame_equal(builder.esm_ds.df, om2_datastore.df)

    def test_init_with_parser(
        self, om2_esmds_path, om2_datastore, intake_esm_kwargs, tmpdir
    ):
        """
        Initialisation with a parser should use the provided parser not instantiate
        the esm åtastore until it's asked for
        """
        dummy_store_path = tmpdir / "dummy_store.icechunk"
        parser = virtualizarr.parsers.hdf.hdf.HDFParser
        builder = IcechunkStoreBuilder(
            om2_esmds_path, intake_esm_kwargs, dummy_store_path, parser=parser
        )

        assert builder._esm_ds is None
        assert isinstance(builder.parser, virtualizarr.parsers.hdf.hdf.HDFParser)

        assert_frame_equal(builder.esm_ds.df, om2_datastore.df)

    @pytest.mark.parametrize(
        "format_val, parser",
        [
            ("netcdf", virtualizarr.parsers.HDFParser),
            ("zarr", virtualizarr.parsers.ZarrParser),
            ("zarr2", virtualizarr.parsers.ZarrParser),
            ("zarr3", virtualizarr.parsers.ZarrParser),
            ("reference", virtualizarr.parsers.KerchunkJSONParser),
        ],
    )
    def test_infer_parser(
        self, om2_esmds_path, intake_esm_kwargs, tmpdir, format_val, parser
    ):
        """
        Mostly a regression test for now.
        """

        dummy_store_path = tmpdir / "dummy_store.icechunk"
        builder = IcechunkStoreBuilder(
            om2_esmds_path, intake_esm_kwargs, dummy_store_path
        )
        from intake_esm.cat import DataFormat

        builder.esm_ds.esmcat.assets.format = DataFormat(format_val)

        inferred_parser = builder._infer_parser()

        assert inferred_parser == parser
