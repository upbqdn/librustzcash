//! Types and functions for building TZE transaction components
#![cfg(feature = "zfuture")]

use std::fmt;

use crate::{
    extensions::transparent::{self as tze, ToPayload},
    transaction::{
        self as tx,
        components::{
            amount::Amount,
            tze::{Authorization, Authorized, Bundle, OutPoint, TzeIn, TzeOut},
        },
    },
};

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    InvalidAmount,
    WitnessModeMismatch(u32, u32),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::InvalidAmount => write!(f, "Invalid amount"),
            Error::WitnessModeMismatch(expected, actual) =>
                write!(f, "TZE witness builder returned a mode that did not match the mode with which the input was initially constructed: expected = {:?}, actual = {:?}", expected, actual),
        }
    }
}

#[allow(clippy::type_complexity)]
pub struct TzeSigner<'a, BuildCtx> {
    prevout: TzeOut,
    builder: Box<dyn FnOnce(&BuildCtx) -> Result<(u32, Vec<u8>), Error> + 'a>,
}

pub struct TzeBuilder<'a, BuildCtx> {
    signers: Vec<TzeSigner<'a, BuildCtx>>,
    vin: Vec<TzeIn<()>>,
    vout: Vec<TzeOut>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Unauthorized;

impl Authorization for Unauthorized {
    type Witness = ();
}

impl<'a, BuildCtx> TzeBuilder<'a, BuildCtx> {
    pub fn empty() -> Self {
        TzeBuilder {
            signers: vec![],
            vin: vec![],
            vout: vec![],
        }
    }

    pub fn add_input<WBuilder, W: ToPayload>(
        &mut self,
        extension_id: u32,
        mode: u32,
        (outpoint, prevout): (OutPoint, TzeOut),
        witness_builder: WBuilder,
    ) where
        WBuilder: 'a + FnOnce(&BuildCtx) -> Result<W, Error>,
    {
        self.vin.push(TzeIn::new(outpoint, extension_id, mode));
        self.signers.push(TzeSigner {
            prevout,
            builder: Box::new(move |ctx| witness_builder(ctx).map(|x| x.to_payload())),
        });
    }

    pub fn add_output<G: ToPayload>(
        &mut self,
        extension_id: u32,
        value: Amount,
        guarded_by: &G,
    ) -> Result<(), Error> {
        if value.is_negative() {
            return Err(Error::InvalidAmount);
        }

        let (mode, payload) = guarded_by.to_payload();
        self.vout.push(TzeOut {
            value,
            precondition: tze::Precondition {
                extension_id,
                mode,
                payload,
            },
        });

        Ok(())
    }

    pub fn value_balance(&self) -> Option<Amount> {
        self.signers
            .iter()
            .map(|s| s.prevout.value)
            .sum::<Option<Amount>>()?
            - self
                .vout
                .iter()
                .map(|tzo| tzo.value)
                .sum::<Option<Amount>>()?
    }

    pub fn build(self) -> (Option<Bundle<Unauthorized>>, Vec<TzeSigner<'a, BuildCtx>>) {
        if self.vin.is_empty() && self.vout.is_empty() {
            (None, vec![])
        } else {
            (
                Some(Bundle {
                    vin: self.vin.clone(),
                    vout: self.vout.clone(),
                    authorization: Unauthorized,
                }),
                self.signers,
            )
        }
    }
}

impl Bundle<Unauthorized> {
    pub fn into_authorized(
        self,
        unauthed_tx: &tx::TransactionData<tx::Unauthorized>,
        signers: Vec<TzeSigner<'_, tx::TransactionData<tx::Unauthorized>>>,
    ) -> Result<Bundle<Authorized>, Error> {
        // Create TZE input witnesses
        let payloads = signers
            .into_iter()
            .zip(self.vin.iter())
            .into_iter()
            .map(|(signer, tzein)| {
                // The witness builder function should have cached/closed over whatever data was
                // necessary for the witness to commit to at the time it was added to the
                // transaction builder; here, it then computes those commitments.
                let (mode, payload) = (signer.builder)(unauthed_tx)?;
                let input_mode = tzein.witness.mode;
                if mode != input_mode {
                    return Err(Error::WitnessModeMismatch(input_mode, mode));
                }

                Ok(tze::AuthData(payload))
            })
            .collect::<Result<Vec<_>, Error>>()?;

        Ok(Bundle {
            vin: self
                .vin
                .into_iter()
                .zip(payloads.into_iter())
                .map(|(tzein, payload)| TzeIn {
                    prevout: tzein.prevout,
                    witness: tze::Witness {
                        extension_id: tzein.witness.extension_id,
                        mode: tzein.witness.mode,
                        payload,
                    },
                })
                .collect(),
            vout: self.vout,
            authorization: Authorized,
        })
    }
}
