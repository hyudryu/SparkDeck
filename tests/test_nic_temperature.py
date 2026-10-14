import tempfile
import unittest
from pathlib import Path, PurePosixPath

from manager import Manager


class NicTemperatureTests(unittest.TestCase):
    """_read_nics surfaces network-card temperature only when a sensor exists."""

    def manager(self) -> Manager:
        return Manager.__new__(Manager)

    @staticmethod
    def pci_device(root: Path, rel: str, *, pci_id: str,
                   pci_class: str = "0x020000", temps: list[str] | None = None) -> Path:
        device = root / rel
        device.mkdir(parents=True)
        (device / "uevent").write_text(f"PCI_ID={pci_id}\n")
        (device / "class").write_text(pci_class + "\n")
        for index, value in enumerate(temps or []):
            hwmon = device / "hwmon" / f"hwmon{index}"
            hwmon.mkdir(parents=True)
            (hwmon / "temp1_input").write_text(value + "\n")
        return device

    def test_connectx_detected_with_named_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.pci_device(
                root, "class/infiniband/mlx5_0/device",
                pci_id="15b3:1021", temps=["54000"],
            )
            self.assertEqual(
                self.manager()._read_nics(root),
                [{"name": "NVIDIA ConnectX-7", "temp": 54.0}],
            )

    def test_unknown_mellanox_card_falls_back_to_connectx_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.pci_device(
                root, "class/net/eth0/device",
                pci_id="15b3:abcd", temps=["61000"],
            )
            self.assertEqual(
                self.manager()._read_nics(root),
                [{"name": "NVIDIA ConnectX", "temp": 61.0}],
            )

    def test_connectx_6_dx_uses_pci_ids_database_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.pci_device(
                root, "class/infiniband/mlx5_0/device",
                pci_id="15b3:101d", temps=["57000"],
            )
            self.assertEqual(
                self.manager()._read_nics(root),
                [{"name": "NVIDIA ConnectX-6 Dx", "temp": 57.0}],
            )

    def test_thunderbolt_class_device_detected_without_netdev(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.pci_device(
                root, "bus/pci/devices/tb-card",
                pci_id="8086:0001", pci_class="0x0c0340", temps=["48000"],
            )
            self.assertEqual(
                self.manager()._read_nics(root),
                [{"name": "Thunderbolt controller", "temp": 48.0}],
            )

    def test_cards_without_temperature_sensor_are_omitted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.pci_device(root, "class/net/eth0/device", pci_id="8086:15f8")
            # Virtual interfaces never surface, even with a sensor.
            self.pci_device(
                root, "class/net/veth12/device", pci_id="15b3:1021", temps=["40000"],
            )
            self.assertEqual(self.manager()._read_nics(root), [])

    def test_generic_card_with_sensor_reports_max_of_valid_sensors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.pci_device(
                root, "class/net/eth0/device", pci_id="1ded:0001",
                temps=["70000", "-55000", "999000"],
            )
            # -55°C is below the plausible floor and 999°C is noise;
            # only the 70°C sensor survives.
            self.assertEqual(
                self.manager()._read_nics(root),
                [{"name": "Network adapter", "temp": 70.0}],
            )

    def test_socket_direct_peers_merge_via_shared_hca_uid(self) -> None:
        # One physical ConnectX-7 presented on two PCIe connections: the
        # verbs devices resolve to different paths but share the HCA uid.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.pci_device(
                root, "class/infiniband/mlx5_0/device",
                pci_id="15b3:1021", temps=["54000"],
            )
            (root / "class/infiniband/mlx5_0/uid").write_text("0x0002c9990000abcd\n")
            self.pci_device(
                root, "class/infiniband/mlx5_1/device",
                pci_id="15b3:1021", temps=["56000"],
            )
            (root / "class/infiniband/mlx5_1/uid").write_text("0x0002c9990000abcd\n")
            self.assertEqual(
                self.manager()._read_nics(root),
                [{"name": "NVIDIA ConnectX-7", "temp": 56.0}],
            )

    def test_distinct_cards_with_different_uids_stay_separate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.pci_device(
                root, "class/infiniband/mlx5_0/device",
                pci_id="15b3:1021", temps=["54000"],
            )
            (root / "class/infiniband/mlx5_0/uid").write_text("0x0002c9000000aaaa\n")
            self.pci_device(
                root, "class/infiniband/mlx5_1/device",
                pci_id="15b3:1021", temps=["61000"],
            )
            (root / "class/infiniband/mlx5_1/uid").write_text("0x0002c9000000bbbb\n")
            self.assertEqual(
                self.manager()._read_nics(root),
                [
                    {"name": "NVIDIA ConnectX-7", "temp": 54.0},
                    {"name": "NVIDIA ConnectX-7", "temp": 61.0},
                ],
            )

    def test_socket_direct_peers_merge_via_shared_sys_image_guid(self) -> None:
        # Stable-ABI device-root sys_image_guid is shared by every port of
        # one physical card, including Socket Direct peers on separate buses.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.pci_device(
                root, "class/infiniband/mlx5_0/device",
                pci_id="15b3:1021", temps=["54000"],
            )
            (root / "class/infiniband/mlx5_0/sys_image_guid").write_text("0x0002c90300abcdef\n")
            self.pci_device(
                root, "class/infiniband/mlx5_1/device",
                pci_id="15b3:1021", temps=["56000"],
            )
            (root / "class/infiniband/mlx5_1/sys_image_guid").write_text("0x0002c90300abcdef\n")
            self.assertEqual(
                self.manager()._read_nics(root),
                [{"name": "NVIDIA ConnectX-7", "temp": 56.0}],
            )

    def test_per_port_sys_image_guid_is_a_fallback_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.pci_device(
                root, "class/infiniband/mlx5_0/device",
                pci_id="15b3:1021", temps=["54000"],
            )
            (root / "class/infiniband/mlx5_0/ports/1").mkdir(parents=True)
            (root / "class/infiniband/mlx5_0/ports/1/sys_image_guid").write_text("0x0002c90300abcdef\n")
            self.pci_device(
                root, "class/infiniband/mlx5_1/device",
                pci_id="15b3:1021", temps=["56000"],
            )
            (root / "class/infiniband/mlx5_1/ports/1").mkdir(parents=True)
            (root / "class/infiniband/mlx5_1/ports/1/sys_image_guid").write_text("0x0002c90300abcdef\n")
            self.assertEqual(
                self.manager()._read_nics(root),
                [{"name": "NVIDIA ConnectX-7", "temp": 56.0}],
            )

    @staticmethod
    def vpd_bytes(serial: str) -> bytes:
        # Standard layout: value-length header (0x80), identifier string
        # (0x82), read-only resource (0x90) carrying the SN keyword, then
        # the 78 78 end marker. Large resources use a 2-byte length.
        field = serial.encode("ascii")
        readonly = b"SN" + bytes([len(field)]) + field + b"\x78\x78"
        ident = b"Mellanox ConnectX"
        return (
            b"\x80\x00\x04\x00\x00\x00\x00"
            + bytes([0x82, 0x00, len(ident)]) + ident
            + bytes([0x90, 0x00, len(readonly)]) + readonly
            + b"\x78\x78"
        )

    def test_socket_direct_merges_via_vpd_serial_without_rdma(self) -> None:
        # No mlx5_ib loaded: /sys/class/infiniband is absent, so the two
        # PCIe connections identify the card through their shared VPD serial.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for iface, temp in (("eth0", "54000"), ("eth1", "56000")):
                device = self.pci_device(
                    root, f"class/net/{iface}/device",
                    pci_id="15b3:1021", temps=[temp],
                )
                (device / "vpd").write_bytes(self.vpd_bytes("MTA1234X"))
            self.assertEqual(
                self.manager()._read_nics(root),
                [{"name": "NVIDIA ConnectX-7", "temp": 56.0}],
            )

    def test_distinct_cards_with_different_vpd_serials_stay_separate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            device0 = self.pci_device(
                root, "class/net/eth0/device",
                pci_id="15b3:1021", temps=["54000"],
            )
            (device0 / "vpd").write_bytes(self.vpd_bytes("MTA1111A"))
            device1 = self.pci_device(
                root, "class/net/eth1/device",
                pci_id="15b3:1021", temps=["61000"],
            )
            (device1 / "vpd").write_bytes(self.vpd_bytes("MTA2222B"))
            self.assertEqual(
                self.manager()._read_nics(root),
                [
                    {"name": "NVIDIA ConnectX-7", "temp": 54.0},
                    {"name": "NVIDIA ConnectX-7", "temp": 61.0},
                ],
            )

    def test_faulted_sensor_channels_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            device = self.pci_device(
                root, "class/net/eth0/device", pci_id="8086:15f8",
            )
            hwmon = device / "hwmon" / "hwmon0"
            hwmon.mkdir(parents=True)
            # The first channel is a stale 45°C sentinel behind a fault flag;
            # only the healthy 70°C channel may be reported.
            (hwmon / "temp1_input").write_text("45000\n")
            (hwmon / "temp1_fault").write_text("1\n")
            (hwmon / "temp2_input").write_text("70000\n")
            (hwmon / "temp2_fault").write_text("0\n")
            self.assertEqual(
                self.manager()._read_nics(root),
                [{"name": "Network adapter", "temp": 70.0}],
            )

    def test_card_with_only_faulted_sensors_is_omitted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            device = self.pci_device(
                root, "class/net/eth0/device", pci_id="8086:15f8",
            )
            hwmon = device / "hwmon" / "hwmon0"
            hwmon.mkdir(parents=True)
            (hwmon / "temp1_input").write_text("60000\n")
            (hwmon / "temp1_fault").write_text("1\n")
            self.assertEqual(self.manager()._read_nics(root), [])

    def test_distinct_cards_with_different_guids_stay_separate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.pci_device(
                root, "class/infiniband/mlx5_0/device",
                pci_id="15b3:1021", temps=["54000"],
            )
            (root / "class/infiniband/mlx5_0/sys_image_guid").write_text("0x0002c9030000aaaa\n")
            self.pci_device(
                root, "class/infiniband/mlx5_1/device",
                pci_id="15b3:1021", temps=["61000"],
            )
            (root / "class/infiniband/mlx5_1/sys_image_guid").write_text("0x0002c9030000bbbb\n")
            self.assertEqual(
                self.manager()._read_nics(root),
                [
                    {"name": "NVIDIA ConnectX-7", "temp": 54.0},
                    {"name": "NVIDIA ConnectX-7", "temp": 61.0},
                ],
            )

    def test_pci_functions_of_one_card_share_a_card_key(self) -> None:
        key = Manager._nic_card_key
        first = PurePosixPath("/sys/devices/pci0000:2a/0000:2a:00.0")
        peer = PurePosixPath("/sys/devices/pci0000:2a/0000:2a:00.1")
        other_card = PurePosixPath("/sys/devices/pci0000:2a/0000:2b:00.0")
        self.assertEqual(key(first), key(peer))
        self.assertNotEqual(key(first), key(other_card))
        # Non-PCI-function paths key verbatim.
        self.assertEqual(key(PurePosixPath("/a/eth0/device")), "/a/eth0/device")

    def test_unreadable_sysfs_returns_no_cards(self) -> None:
        self.assertEqual(self.manager()._read_nics(Path("/definitely/not/sys")), [])


if __name__ == "__main__":
    unittest.main()
