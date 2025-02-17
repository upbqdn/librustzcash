mod add_transaction_views;
mod addresses_table;
mod initial_setup;
mod ufvk_support;
mod utxos_table;

use schemer_rusqlite::RusqliteMigration;
use secrecy::SecretVec;
use zcash_primitives::consensus;

use super::WalletMigrationError;

pub(super) fn all_migrations<P: consensus::Parameters + 'static>(
    params: &P,
    seed: Option<SecretVec<u8>>,
) -> Vec<Box<dyn RusqliteMigration<Error = WalletMigrationError>>> {
    vec![
        Box::new(initial_setup::Migration {}),
        Box::new(utxos_table::Migration {}),
        Box::new(ufvk_support::Migration {
            params: params.clone(),
            seed,
        }),
        Box::new(addresses_table::Migration {
            params: params.clone(),
        }),
        Box::new(add_transaction_views::Migration {
            params: params.clone(),
        }),
    ]
}
