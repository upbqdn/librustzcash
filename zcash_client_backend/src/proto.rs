//! Generated code for handling light client protobuf structs.

use group::ff::PrimeField;

use zcash_primitives::{
    block::{BlockHash, BlockHeader},
    consensus::BlockHeight,
    sapling::Nullifier,
    transaction::{components::sapling, TxId},
};

use zcash_note_encryption::{EphemeralKeyBytes, COMPACT_NOTE_SIZE};

pub mod compact_formats;

impl compact_formats::CompactBlock {
    /// Returns the [`BlockHash`] for this block.
    ///
    /// # Panics
    ///
    /// This function will panic if [`CompactBlock.header`] is not set and
    /// [`CompactBlock.hash`] is not exactly 32 bytes.
    ///
    /// [`CompactBlock.header`]: #structfield.header
    /// [`CompactBlock.hash`]: #structfield.hash
    pub fn hash(&self) -> BlockHash {
        if let Some(header) = self.header() {
            header.hash()
        } else {
            BlockHash::from_slice(&self.hash)
        }
    }

    /// Returns the [`BlockHash`] for this block's parent.
    ///
    /// # Panics
    ///
    /// This function will panic if [`CompactBlock.header`] is not set and
    /// [`CompactBlock.prevHash`] is not exactly 32 bytes.
    ///
    /// [`CompactBlock.header`]: #structfield.header
    /// [`CompactBlock.prevHash`]: #structfield.prevHash
    pub fn prev_hash(&self) -> BlockHash {
        if let Some(header) = self.header() {
            header.prev_block
        } else {
            BlockHash::from_slice(&self.prevHash)
        }
    }

    /// Returns the [`BlockHeight`] value for this block
    ///
    /// # Panics
    ///
    /// This function will panic if [`CompactBlock.height`] is not
    /// representable within a u32.
    pub fn height(&self) -> BlockHeight {
        self.height.try_into().unwrap()
    }

    /// Returns the [`BlockHeader`] for this block if present.
    ///
    /// A convenience method that parses [`CompactBlock.header`] if present.
    ///
    /// [`CompactBlock.header`]: #structfield.header
    pub fn header(&self) -> Option<BlockHeader> {
        if self.header.is_empty() {
            None
        } else {
            BlockHeader::read(&self.header[..]).ok()
        }
    }
}

impl compact_formats::CompactTx {
    /// Returns the transaction Id
    pub fn txid(&self) -> TxId {
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&self.hash);
        TxId::from_bytes(hash)
    }
}

impl compact_formats::CompactSaplingOutput {
    /// Returns the note commitment for this output.
    ///
    /// A convenience method that parses [`CompactOutput.cmu`].
    ///
    /// [`CompactOutput.cmu`]: #structfield.cmu
    pub fn cmu(&self) -> Result<bls12_381::Scalar, ()> {
        let mut repr = [0; 32];
        repr.as_mut().copy_from_slice(&self.cmu[..]);
        Option::from(bls12_381::Scalar::from_repr(repr)).ok_or(())
    }

    /// Returns the ephemeral public key for this output.
    ///
    /// A convenience method that parses [`CompactOutput.epk`].
    ///
    /// [`CompactOutput.epk`]: #structfield.epk
    pub fn ephemeral_key(&self) -> Result<EphemeralKeyBytes, ()> {
        self.ephemeralKey[..]
            .try_into()
            .map(EphemeralKeyBytes)
            .map_err(|_| ())
    }
}

impl<A: sapling::Authorization> From<sapling::OutputDescription<A>>
    for compact_formats::CompactSaplingOutput
{
    fn from(out: sapling::OutputDescription<A>) -> compact_formats::CompactSaplingOutput {
        let mut result = compact_formats::CompactSaplingOutput::new();
        result.set_cmu(out.cmu.to_repr().to_vec());
        result.set_ephemeralKey(out.ephemeral_key.as_ref().to_vec());
        result.set_ciphertext(out.enc_ciphertext[..COMPACT_NOTE_SIZE].to_vec());
        result
    }
}

impl TryFrom<compact_formats::CompactSaplingOutput> for sapling::CompactOutputDescription {
    type Error = ();

    fn try_from(value: compact_formats::CompactSaplingOutput) -> Result<Self, Self::Error> {
        Ok(sapling::CompactOutputDescription {
            cmu: value.cmu()?,
            ephemeral_key: value.ephemeral_key()?,
            enc_ciphertext: value.ciphertext.try_into().map_err(|_| ())?,
        })
    }
}

impl compact_formats::CompactSaplingSpend {
    pub fn nf(&self) -> Result<Nullifier, ()> {
        Nullifier::from_slice(&self.nf).map_err(|_| ())
    }
}
