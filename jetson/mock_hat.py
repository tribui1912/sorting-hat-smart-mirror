#!/usr/bin/env python3
"""Fake hat firmware for testing send_audio.py without hardware.

Speaks the same TCP protocol as hat_firmware.ino (PING/CLIP/PLAY),
reports what it received, and can dump the PCM to a file so you can
check the stream is intact:
  python jetson/mock_hat.py [port] [--dump out.pcm]
"""

import socket
import sys

RATE = 22050
PORT = 5005


def main():
    port = PORT
    dump = None
    args = sys.argv[1:]
    if "--dump" in args:
        i = args.index("--dump")
        dump = args[i + 1]
        del args[i:i + 2]
    if args:
        port = int(args[0])

    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(1)
    print(f"mock hat listening on :{port} (ctrl-C to stop)")

    while True:
        c, addr = srv.accept()
        print(f"client {addr[0]}")
        f = c.makefile("rb")
        try:
            for line in f:
                cmd = line.decode(errors="replace").strip()
                if cmd == "PING":
                    c.sendall(b"PONG\n")
                elif cmd == "CLIP":
                    print("  -> would play embedded clip")
                    c.sendall(b"DONE\n")
                elif cmd.startswith("PLAY "):
                    n = int(cmd.split()[1])
                    c.sendall(b"OK\n")
                    got, out = 0, b"" if dump else None
                    while got < n:
                        d = f.read(min(65536, n - got))
                        if not d:
                            print(f"  !! stream cut off at {got}/{n}")
                            break
                        got += len(d)
                        if dump:
                            out += d
                    print(f"  -> PLAY {n} B ({n / (RATE * 2):.2f} s): "
                          f"received {got} {'OK' if got == n else 'SHORT'}")
                    if dump and out:
                        open(dump, "wb").write(out)
                        print(f"  -> wrote {dump}")
                    c.sendall(b"DONE\n")
                else:
                    c.sendall(b"ERR unknown command\n")
                    break
        finally:
            c.close()
            print("  client gone")


if __name__ == "__main__":
    main()
