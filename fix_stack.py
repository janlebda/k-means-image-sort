import os, struct, glob

def remove_execstack():
    prefix = os.environ.get('CONDA_PREFIX', '')
    if not prefix:
        print("Uruchom skrypt wewnątrz środowiska Conda!")
        return

    print("Szukanie zablokowanych bibliotek...")
    patterns = [
        f"{prefix}/lib/libtensorflow*.so*",
        f"{prefix}/lib/python*/site-packages/tensorflow/**/*.so*",
        f"{prefix}/lib/python*/site-packages/nvidia/**/*.so*"  # NOWE: leczy pliki CUDA od Nvidii
    ]
    
    files = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))

    patched = 0
    for fpath in set(files):
        if os.path.islink(fpath) or not os.path.isfile(fpath): continue
        try:
            with open(fpath, 'r+b') as f:
                if f.read(4) != b'\x7fELF': continue
                f.seek(4)
                if f.read(1)[0] != 2: continue

                f.seek(32)
                e_phoff = struct.unpack('<Q', f.read(8))[0]
                f.seek(54)
                e_phentsize, e_phnum = struct.unpack('<HH', f.read(4))
                
                for i in range(e_phnum):
                    offset = e_phoff + i * e_phentsize
                    f.seek(offset)
                    ptype, pflags = struct.unpack('<II', f.read(8))
                    
                    if ptype == 0x6474e551:
                        if pflags & 1: 
                            f.seek(offset + 4)
                            f.write(struct.pack('<I', pflags & ~1))
                            print(f"[+] Uleczono plik: {os.path.basename(fpath)}")
                            patched += 1
                        break
        except Exception: pass

    if patched == 0:
        print("Brak plików do poprawy.")
    else:
        print(f"Gotowe! Oczyszczono {patched} plików z flagi execstack.")

if __name__ == '__main__':
    remove_execstack()