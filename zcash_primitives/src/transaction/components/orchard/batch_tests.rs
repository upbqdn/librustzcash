use alloc::vec::Vec;
use corez::io::{self, Cursor, Write as _};
use group::{Group as _, GroupEncoding as _};
use pasta_curves::pallas;
use zcash_encoding::CompactSize;

use super::{read_action_without_auth, read_actions_without_auth, ACTION_SIZE};

fn action_bytes() -> [u8; ACTION_SIZE] {
    let mut bytes = [0; ACTION_SIZE];
    let point = pallas::Point::generator().to_bytes();
    bytes[..32].copy_from_slice(&point); // cv
    bytes[32..64].copy_from_slice(&[1; 32]); // nf
    bytes[64..96].copy_from_slice(&point); // rk
    bytes[96..128].copy_from_slice(&[2; 32]); // cmx
    bytes[128..160].copy_from_slice(&point); // epk
    read_action_without_auth(bytes.as_slice())
        .expect("a generated action has valid field encodings");
    bytes
}

fn wire(actions: &[[u8; ACTION_SIZE]]) -> Vec<u8> {
    let mut bytes = Vec::new();
    CompactSize::write(&mut bytes, actions.len()).unwrap();
    for action in actions {
        bytes.write_all(action).unwrap();
    }
    bytes
}

fn check(wire: &[u8]) {
    let mut original = Cursor::new(wire);
    let count: usize = CompactSize::read_t(&mut original).unwrap();
    let serial: io::Result<Vec<_>> = (0..count)
        .map(|_| read_action_without_auth(&mut original))
        .collect();
    let batch = read_actions_without_auth(Cursor::new(wire));
    match (serial, batch) {
        (Ok(serial), Ok(batch)) => {
            assert_eq!(serial.len(), batch.len());
            for (serial, batch) in serial.iter().zip(&batch) {
                assert_eq!(serial.cv_net().to_bytes(), batch.cv_net().to_bytes());
                assert_eq!(serial.encrypted_note().epk_bytes, batch.encrypted_note().epk_bytes);
                assert_eq!(serial.rk().to_bytes(), batch.rk().to_bytes());
            }
        }
        (Err(serial), Err(batch)) => {
            assert_eq!(serial.kind(), batch.kind());
            assert_eq!(serial.to_string(), batch.to_string());
        }
        (serial, batch) => panic!("different acceptance: {:?} vs {:?}", serial, batch),
    }
}

#[test]
fn batch_preserves_validity_first_error_and_truncated_input_precedence() {
    let valid = action_bytes();
    let mut cv = valid;
    cv[..32].fill(0xff);
    let mut nf = valid;
    nf[32..64].fill(0xff);
    let mut rk = valid;
    rk[64..96].fill(0xff);
    let mut cmx = valid;
    cmx[96..128].fill(0xff);
    let mut epk = valid;
    epk[128..160].fill(0xff);
    let mut identity_rk = valid;
    identity_rk[64..96].fill(0);
    let mut identity_epk = valid;
    identity_epk[128..160].fill(0);
    let mut identity_cv = valid;
    identity_cv[..32].fill(0);

    for actions in [
        [valid, valid, valid],
        [identity_cv, valid, valid],
        [valid, cv, valid],
        [valid, nf, valid],
        [valid, rk, valid],
        [valid, cmx, valid],
        [valid, epk, valid],
        [valid, identity_rk, valid],
        [valid, identity_epk, valid],
        [epk, cv, valid],
        [cv, rk, epk],
        [identity_rk, epk, cv],
    ] {
        check(&wire(&actions));
    }
    let mut cv_before_nf = cv;
    cv_before_nf[32..64].fill(0xff);
    check(&wire(&[cv_before_nf, valid]));
    let mut rk_before_cmx = identity_rk;
    rk_before_cmx[96..128].fill(0xff);
    check(&wire(&[rk_before_cmx, valid]));

    // Exercise a full eight-point chunk followed by another chunk.
    let mut actions = [valid; 9];
    check(&wire(&actions));
    actions[4] = epk;
    actions[5] = cv;
    check(&wire(&actions));
    actions[4] = valid;
    actions[8] = identity_epk;
    check(&wire(&actions));
    let mut truncated = wire(&actions);
    truncated.truncate(1 + 8 * ACTION_SIZE + 100);
    check(&truncated);

    for first in [valid, cv, epk, identity_rk] {
        for bytes_of_second in [0, 16, 32, 65, 96, 128, 160, 500, ACTION_SIZE - 1] {
            let mut truncated = wire(&[first, valid]);
            truncated.truncate(1 + ACTION_SIZE + bytes_of_second);
            check(&truncated);
        }
    }
}
