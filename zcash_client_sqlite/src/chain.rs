//! Functions for enforcing chain validity and handling chain reorgs.
use protobuf::Message;

use rusqlite::params;

use zcash_primitives::consensus::BlockHeight;

use zcash_client_backend::{data_api::error::Error, proto::compact_formats::CompactBlock};

use crate::{error::SqliteClientError, BlockDb};

#[cfg(feature = "unstable")]
use {
    crate::{BlockHash, FsBlockDb},
    rusqlite::Connection,
    std::fs::File,
    std::io::BufReader,
    std::path::{Path, PathBuf},
};

pub mod init;
pub mod migrations;

struct CompactBlockRow {
    height: BlockHeight,
    data: Vec<u8>,
}

/// Implements a traversal of `limit` blocks of the block cache database.
///
/// Starting at the next block above `last_scanned_height`, the `with_row` callback is invoked with
/// each block retrieved from the backing store. If the `limit` value provided is `None`, all
/// blocks are traversed up to the maximum height.
pub(crate) fn blockdb_with_blocks<F>(
    cache: &BlockDb,
    last_scanned_height: BlockHeight,
    limit: Option<u32>,
    mut with_row: F,
) -> Result<(), SqliteClientError>
where
    F: FnMut(CompactBlock) -> Result<(), SqliteClientError>,
{
    // Fetch the CompactBlocks we need to scan
    let mut stmt_blocks = cache.0.prepare(
        "SELECT height, data FROM compactblocks WHERE height > ? ORDER BY height ASC LIMIT ?",
    )?;

    let rows = stmt_blocks.query_map(
        params![
            u32::from(last_scanned_height),
            limit.unwrap_or(u32::max_value()),
        ],
        |row| {
            Ok(CompactBlockRow {
                height: BlockHeight::from_u32(row.get(0)?),
                data: row.get(1)?,
            })
        },
    )?;

    for row_result in rows {
        let cbr = row_result?;
        let block: CompactBlock = Message::parse_from_bytes(&cbr.data).map_err(Error::from)?;

        if block.height() != cbr.height {
            return Err(SqliteClientError::CorruptedData(format!(
                "Block height {} did not match row's height field value {}",
                block.height(),
                cbr.height
            )));
        }

        with_row(block)?;
    }

    Ok(())
}

/// Data structure representing a row in the block metadata database.
#[cfg(feature = "unstable")]
pub struct BlockMeta {
    pub height: BlockHeight,
    pub block_hash: BlockHash,
    pub block_time: u32,
    pub sapling_outputs_count: u32,
    pub orchard_actions_count: u32,
}

#[cfg(feature = "unstable")]
impl BlockMeta {
    pub fn block_file_path<P: AsRef<Path>>(&self, blocks_dir: &P) -> PathBuf {
        blocks_dir.as_ref().join(Path::new(&format!(
            "{}-{}-compactblock",
            self.height, self.block_hash
        )))
    }
}

/// Inserts a batch of rows into the block metadata database.
#[cfg(feature = "unstable")]
pub(crate) fn blockmetadb_insert(
    conn: &Connection,
    block_meta: &[BlockMeta],
) -> Result<(), rusqlite::Error> {
    let mut stmt_insert = conn.prepare(
        "INSERT INTO compactblocks_meta (height, blockhash, time, sapling_outputs_count, orchard_actions_count)
        VALUES (?, ?, ?, ?, ?)"
    )?;

    conn.execute("BEGIN IMMEDIATE", [])?;
    let result = block_meta
        .iter()
        .map(|m| {
            stmt_insert.execute(params![
                u32::from(m.height),
                &m.block_hash.0[..],
                m.block_time,
                m.sapling_outputs_count,
                m.orchard_actions_count,
            ])
        })
        .collect::<Result<Vec<_>, _>>();
    match result {
        Ok(_) => {
            conn.execute("COMMIT", [])?;
            Ok(())
        }
        Err(error) => {
            match conn.execute("ROLLBACK", []) {
                Ok(_) => Err(error),
                Err(e) =>
                    // Panicking here is probably the right thing to do, because it
                    // means the database is corrupt.
                    panic!(
                        "Rollback failed with error {} while attempting to recover from error {}; database is likely corrupt.",
                        e,
                        error
                    )
            }
        }
    }
}

#[cfg(feature = "unstable")]
pub(crate) fn blockmetadb_get_max_cached_height(
    conn: &Connection,
) -> Result<Option<BlockHeight>, rusqlite::Error> {
    conn.query_row("SELECT MAX(height) FROM compactblocks_meta", [], |row| {
        // `SELECT MAX(_)` will always return a row, but it will return `null` if the
        // table is empty, which has no integer type. We handle the optionality here.
        let h: Option<u32> = row.get(0)?;
        Ok(h.map(BlockHeight::from))
    })
}

/// Implements a traversal of `limit` blocks of the filesystem-backed
/// block cache.
///
/// Starting at the next block height above `last_scanned_height`, the `with_row` callback is
/// invoked with each block retrieved from the backing store. If the `limit` value provided is
/// `None`, all blocks are traversed up to the maximum height for which metadata is available.
#[cfg(feature = "unstable")]
pub(crate) fn fsblockdb_with_blocks<F>(
    cache: &FsBlockDb,
    last_scanned_height: BlockHeight,
    limit: Option<u32>,
    mut with_block: F,
) -> Result<(), SqliteClientError>
where
    F: FnMut(CompactBlock) -> Result<(), SqliteClientError>,
{
    // Fetch the CompactBlocks we need to scan
    let mut stmt_blocks = cache.conn.prepare(
        "SELECT height, blockhash, time, sapling_outputs_count, orchard_actions_count
         FROM compactblocks_meta
         WHERE height > ?
         ORDER BY height ASC LIMIT ?",
    )?;

    let rows = stmt_blocks.query_map(
        params![
            u32::from(last_scanned_height),
            limit.unwrap_or(u32::max_value()),
        ],
        |row| {
            Ok(BlockMeta {
                height: BlockHeight::from_u32(row.get(0)?),
                block_hash: BlockHash::from_slice(&row.get::<_, Vec<_>>(1)?),
                block_time: row.get(2)?,
                sapling_outputs_count: row.get(3)?,
                orchard_actions_count: row.get(4)?,
            })
        },
    )?;

    for row_result in rows {
        let cbr = row_result?;
        let block_file = File::open(cbr.block_file_path(&cache.blocks_dir))?;
        let mut buf_reader = BufReader::new(block_file);

        let block: CompactBlock =
            Message::parse_from_reader(&mut buf_reader).map_err(Error::from)?;

        if block.height() != cbr.height {
            return Err(SqliteClientError::CorruptedData(format!(
                "Block height {} did not match row's height field value {}",
                block.height(),
                cbr.height
            )));
        }

        with_block(block)?;
    }

    Ok(())
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use secrecy::Secret;
    use tempfile::NamedTempFile;

    use zcash_primitives::{
        block::BlockHash, transaction::components::Amount, zip32::ExtendedSpendingKey,
    };

    use zcash_client_backend::data_api::WalletRead;
    use zcash_client_backend::data_api::{
        chain::{scan_cached_blocks, validate_chain},
        error::{ChainInvalid, Error},
    };

    use crate::{
        chain::init::init_cache_database,
        error::SqliteClientError,
        tests::{
            self, fake_compact_block, fake_compact_block_spending, init_test_accounts_table,
            insert_into_cache, sapling_activation_height,
        },
        wallet::{get_balance, init::init_wallet_db, rewind_to_height},
        AccountId, BlockDb, NoteId, WalletDb,
    };

    #[test]
    fn valid_chain_states() {
        let cache_file = NamedTempFile::new().unwrap();
        let db_cache = BlockDb::for_path(cache_file.path()).unwrap();
        init_cache_database(&db_cache).unwrap();

        let data_file = NamedTempFile::new().unwrap();
        let mut db_data = WalletDb::for_path(data_file.path(), tests::network()).unwrap();
        init_wallet_db(&mut db_data, Some(Secret::new(vec![]))).unwrap();

        // Add an account to the wallet
        let (dfvk, _taddr) = init_test_accounts_table(&db_data);

        // Empty chain should be valid
        validate_chain(
            &tests::network(),
            &db_cache,
            db_data.get_max_height_hash().unwrap(),
        )
        .unwrap();

        // Create a fake CompactBlock sending value to the address
        let (cb, _) = fake_compact_block(
            sapling_activation_height(),
            BlockHash([0; 32]),
            &dfvk,
            Amount::from_u64(5).unwrap(),
        );
        insert_into_cache(&db_cache, &cb);

        // Cache-only chain should be valid
        validate_chain(
            &tests::network(),
            &db_cache,
            db_data.get_max_height_hash().unwrap(),
        )
        .unwrap();

        // Scan the cache
        let mut db_write = db_data.get_update_ops().unwrap();
        scan_cached_blocks(&tests::network(), &db_cache, &mut db_write, None).unwrap();

        // Data-only chain should be valid
        validate_chain(
            &tests::network(),
            &db_cache,
            db_data.get_max_height_hash().unwrap(),
        )
        .unwrap();

        // Create a second fake CompactBlock sending more value to the address
        let (cb2, _) = fake_compact_block(
            sapling_activation_height() + 1,
            cb.hash(),
            &dfvk,
            Amount::from_u64(7).unwrap(),
        );
        insert_into_cache(&db_cache, &cb2);

        // Data+cache chain should be valid
        validate_chain(
            &tests::network(),
            &db_cache,
            db_data.get_max_height_hash().unwrap(),
        )
        .unwrap();

        // Scan the cache again
        scan_cached_blocks(&tests::network(), &db_cache, &mut db_write, None).unwrap();

        // Data-only chain should be valid
        validate_chain(
            &tests::network(),
            &db_cache,
            db_data.get_max_height_hash().unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn invalid_chain_cache_disconnected() {
        let cache_file = NamedTempFile::new().unwrap();
        let db_cache = BlockDb::for_path(cache_file.path()).unwrap();
        init_cache_database(&db_cache).unwrap();

        let data_file = NamedTempFile::new().unwrap();
        let mut db_data = WalletDb::for_path(data_file.path(), tests::network()).unwrap();
        init_wallet_db(&mut db_data, Some(Secret::new(vec![]))).unwrap();

        // Add an account to the wallet
        let (dfvk, _taddr) = init_test_accounts_table(&db_data);

        // Create some fake CompactBlocks
        let (cb, _) = fake_compact_block(
            sapling_activation_height(),
            BlockHash([0; 32]),
            &dfvk,
            Amount::from_u64(5).unwrap(),
        );
        let (cb2, _) = fake_compact_block(
            sapling_activation_height() + 1,
            cb.hash(),
            &dfvk,
            Amount::from_u64(7).unwrap(),
        );
        insert_into_cache(&db_cache, &cb);
        insert_into_cache(&db_cache, &cb2);

        // Scan the cache
        let mut db_write = db_data.get_update_ops().unwrap();
        scan_cached_blocks(&tests::network(), &db_cache, &mut db_write, None).unwrap();

        // Data-only chain should be valid
        validate_chain(
            &tests::network(),
            &db_cache,
            db_data.get_max_height_hash().unwrap(),
        )
        .unwrap();

        // Create more fake CompactBlocks that don't connect to the scanned ones
        let (cb3, _) = fake_compact_block(
            sapling_activation_height() + 2,
            BlockHash([1; 32]),
            &dfvk,
            Amount::from_u64(8).unwrap(),
        );
        let (cb4, _) = fake_compact_block(
            sapling_activation_height() + 3,
            cb3.hash(),
            &dfvk,
            Amount::from_u64(3).unwrap(),
        );
        insert_into_cache(&db_cache, &cb3);
        insert_into_cache(&db_cache, &cb4);

        // Data+cache chain should be invalid at the data/cache boundary
        match validate_chain(
            &tests::network(),
            &db_cache,
            db_data.get_max_height_hash().unwrap(),
        ) {
            Err(SqliteClientError::BackendError(Error::InvalidChain(lower_bound, _))) => {
                assert_eq!(lower_bound, sapling_activation_height() + 2)
            }
            _ => panic!(),
        }
    }

    #[test]
    fn invalid_chain_cache_reorg() {
        let cache_file = NamedTempFile::new().unwrap();
        let db_cache = BlockDb::for_path(cache_file.path()).unwrap();
        init_cache_database(&db_cache).unwrap();

        let data_file = NamedTempFile::new().unwrap();
        let mut db_data = WalletDb::for_path(data_file.path(), tests::network()).unwrap();
        init_wallet_db(&mut db_data, Some(Secret::new(vec![]))).unwrap();

        // Add an account to the wallet
        let (dfvk, _taddr) = init_test_accounts_table(&db_data);

        // Create some fake CompactBlocks
        let (cb, _) = fake_compact_block(
            sapling_activation_height(),
            BlockHash([0; 32]),
            &dfvk,
            Amount::from_u64(5).unwrap(),
        );
        let (cb2, _) = fake_compact_block(
            sapling_activation_height() + 1,
            cb.hash(),
            &dfvk,
            Amount::from_u64(7).unwrap(),
        );
        insert_into_cache(&db_cache, &cb);
        insert_into_cache(&db_cache, &cb2);

        // Scan the cache
        let mut db_write = db_data.get_update_ops().unwrap();
        scan_cached_blocks(&tests::network(), &db_cache, &mut db_write, None).unwrap();

        // Data-only chain should be valid
        validate_chain(
            &tests::network(),
            &db_cache,
            db_data.get_max_height_hash().unwrap(),
        )
        .unwrap();

        // Create more fake CompactBlocks that contain a reorg
        let (cb3, _) = fake_compact_block(
            sapling_activation_height() + 2,
            cb2.hash(),
            &dfvk,
            Amount::from_u64(8).unwrap(),
        );
        let (cb4, _) = fake_compact_block(
            sapling_activation_height() + 3,
            BlockHash([1; 32]),
            &dfvk,
            Amount::from_u64(3).unwrap(),
        );
        insert_into_cache(&db_cache, &cb3);
        insert_into_cache(&db_cache, &cb4);

        // Data+cache chain should be invalid inside the cache
        match validate_chain(
            &tests::network(),
            &db_cache,
            db_data.get_max_height_hash().unwrap(),
        ) {
            Err(SqliteClientError::BackendError(Error::InvalidChain(lower_bound, _))) => {
                assert_eq!(lower_bound, sapling_activation_height() + 3)
            }
            _ => panic!(),
        }
    }

    #[test]
    fn data_db_rewinding() {
        let cache_file = NamedTempFile::new().unwrap();
        let db_cache = BlockDb::for_path(cache_file.path()).unwrap();
        init_cache_database(&db_cache).unwrap();

        let data_file = NamedTempFile::new().unwrap();
        let mut db_data = WalletDb::for_path(data_file.path(), tests::network()).unwrap();
        init_wallet_db(&mut db_data, Some(Secret::new(vec![]))).unwrap();

        // Add an account to the wallet
        let (dfvk, _taddr) = init_test_accounts_table(&db_data);

        // Account balance should be zero
        assert_eq!(
            get_balance(&db_data, AccountId::from(0)).unwrap(),
            Amount::zero()
        );

        // Create fake CompactBlocks sending value to the address
        let value = Amount::from_u64(5).unwrap();
        let value2 = Amount::from_u64(7).unwrap();
        let (cb, _) = fake_compact_block(
            sapling_activation_height(),
            BlockHash([0; 32]),
            &dfvk,
            value,
        );

        let (cb2, _) =
            fake_compact_block(sapling_activation_height() + 1, cb.hash(), &dfvk, value2);
        insert_into_cache(&db_cache, &cb);
        insert_into_cache(&db_cache, &cb2);

        // Scan the cache
        let mut db_write = db_data.get_update_ops().unwrap();
        scan_cached_blocks(&tests::network(), &db_cache, &mut db_write, None).unwrap();

        // Account balance should reflect both received notes
        assert_eq!(
            get_balance(&db_data, AccountId::from(0)).unwrap(),
            (value + value2).unwrap()
        );

        // "Rewind" to height of last scanned block
        rewind_to_height(&db_data, sapling_activation_height() + 1).unwrap();

        // Account balance should be unaltered
        assert_eq!(
            get_balance(&db_data, AccountId::from(0)).unwrap(),
            (value + value2).unwrap()
        );

        // Rewind so that one block is dropped
        rewind_to_height(&db_data, sapling_activation_height()).unwrap();

        // Account balance should only contain the first received note
        assert_eq!(get_balance(&db_data, AccountId::from(0)).unwrap(), value);

        // Scan the cache again
        scan_cached_blocks(&tests::network(), &db_cache, &mut db_write, None).unwrap();

        // Account balance should again reflect both received notes
        assert_eq!(
            get_balance(&db_data, AccountId::from(0)).unwrap(),
            (value + value2).unwrap()
        );
    }

    #[test]
    fn scan_cached_blocks_requires_sequential_blocks() {
        let cache_file = NamedTempFile::new().unwrap();
        let db_cache = BlockDb::for_path(cache_file.path()).unwrap();
        init_cache_database(&db_cache).unwrap();

        let data_file = NamedTempFile::new().unwrap();
        let mut db_data = WalletDb::for_path(data_file.path(), tests::network()).unwrap();
        init_wallet_db(&mut db_data, Some(Secret::new(vec![]))).unwrap();

        // Add an account to the wallet
        let (dfvk, _taddr) = init_test_accounts_table(&db_data);

        // Create a block with height SAPLING_ACTIVATION_HEIGHT
        let value = Amount::from_u64(50000).unwrap();
        let (cb1, _) = fake_compact_block(
            sapling_activation_height(),
            BlockHash([0; 32]),
            &dfvk,
            value,
        );
        insert_into_cache(&db_cache, &cb1);
        let mut db_write = db_data.get_update_ops().unwrap();
        scan_cached_blocks(&tests::network(), &db_cache, &mut db_write, None).unwrap();
        assert_eq!(get_balance(&db_data, AccountId::from(0)).unwrap(), value);

        // We cannot scan a block of height SAPLING_ACTIVATION_HEIGHT + 2 next
        let (cb2, _) =
            fake_compact_block(sapling_activation_height() + 1, cb1.hash(), &dfvk, value);
        let (cb3, _) =
            fake_compact_block(sapling_activation_height() + 2, cb2.hash(), &dfvk, value);
        insert_into_cache(&db_cache, &cb3);
        match scan_cached_blocks(&tests::network(), &db_cache, &mut db_write, None) {
            Err(SqliteClientError::BackendError(e)) => {
                assert_eq!(
                    e.to_string(),
                    ChainInvalid::block_height_discontinuity::<NoteId>(
                        sapling_activation_height() + 1,
                        sapling_activation_height() + 2
                    )
                    .to_string()
                );
            }
            Ok(_) | Err(_) => panic!("Should have failed"),
        }

        // If we add a block of height SAPLING_ACTIVATION_HEIGHT + 1, we can now scan both
        insert_into_cache(&db_cache, &cb2);
        scan_cached_blocks(&tests::network(), &db_cache, &mut db_write, None).unwrap();
        assert_eq!(
            get_balance(&db_data, AccountId::from(0)).unwrap(),
            Amount::from_u64(150_000).unwrap()
        );
    }

    #[test]
    fn scan_cached_blocks_finds_received_notes() {
        let cache_file = NamedTempFile::new().unwrap();
        let db_cache = BlockDb::for_path(cache_file.path()).unwrap();
        init_cache_database(&db_cache).unwrap();

        let data_file = NamedTempFile::new().unwrap();
        let mut db_data = WalletDb::for_path(data_file.path(), tests::network()).unwrap();
        init_wallet_db(&mut db_data, Some(Secret::new(vec![]))).unwrap();

        // Add an account to the wallet
        let (dfvk, _taddr) = init_test_accounts_table(&db_data);

        // Account balance should be zero
        assert_eq!(
            get_balance(&db_data, AccountId::from(0)).unwrap(),
            Amount::zero()
        );

        // Create a fake CompactBlock sending value to the address
        let value = Amount::from_u64(5).unwrap();
        let (cb, _) = fake_compact_block(
            sapling_activation_height(),
            BlockHash([0; 32]),
            &dfvk,
            value,
        );
        insert_into_cache(&db_cache, &cb);

        // Scan the cache
        let mut db_write = db_data.get_update_ops().unwrap();
        scan_cached_blocks(&tests::network(), &db_cache, &mut db_write, None).unwrap();

        // Account balance should reflect the received note
        assert_eq!(get_balance(&db_data, AccountId::from(0)).unwrap(), value);

        // Create a second fake CompactBlock sending more value to the address
        let value2 = Amount::from_u64(7).unwrap();
        let (cb2, _) =
            fake_compact_block(sapling_activation_height() + 1, cb.hash(), &dfvk, value2);
        insert_into_cache(&db_cache, &cb2);

        // Scan the cache again
        scan_cached_blocks(&tests::network(), &db_cache, &mut db_write, None).unwrap();

        // Account balance should reflect both received notes
        assert_eq!(
            get_balance(&db_data, AccountId::from(0)).unwrap(),
            (value + value2).unwrap()
        );
    }

    #[test]
    fn scan_cached_blocks_finds_change_notes() {
        let cache_file = NamedTempFile::new().unwrap();
        let db_cache = BlockDb::for_path(cache_file.path()).unwrap();
        init_cache_database(&db_cache).unwrap();

        let data_file = NamedTempFile::new().unwrap();
        let mut db_data = WalletDb::for_path(data_file.path(), tests::network()).unwrap();
        init_wallet_db(&mut db_data, Some(Secret::new(vec![]))).unwrap();

        // Add an account to the wallet
        let (dfvk, _taddr) = init_test_accounts_table(&db_data);

        // Account balance should be zero
        assert_eq!(
            get_balance(&db_data, AccountId::from(0)).unwrap(),
            Amount::zero()
        );

        // Create a fake CompactBlock sending value to the address
        let value = Amount::from_u64(5).unwrap();
        let (cb, nf) = fake_compact_block(
            sapling_activation_height(),
            BlockHash([0; 32]),
            &dfvk,
            value,
        );
        insert_into_cache(&db_cache, &cb);

        // Scan the cache
        let mut db_write = db_data.get_update_ops().unwrap();
        scan_cached_blocks(&tests::network(), &db_cache, &mut db_write, None).unwrap();

        // Account balance should reflect the received note
        assert_eq!(get_balance(&db_data, AccountId::from(0)).unwrap(), value);

        // Create a second fake CompactBlock spending value from the address
        let extsk2 = ExtendedSpendingKey::master(&[0]);
        let to2 = extsk2.default_address().1;
        let value2 = Amount::from_u64(2).unwrap();
        insert_into_cache(
            &db_cache,
            &fake_compact_block_spending(
                sapling_activation_height() + 1,
                cb.hash(),
                (nf, value),
                &dfvk,
                to2,
                value2,
            ),
        );

        // Scan the cache again
        scan_cached_blocks(&tests::network(), &db_cache, &mut db_write, None).unwrap();

        // Account balance should equal the change
        assert_eq!(
            get_balance(&db_data, AccountId::from(0)).unwrap(),
            (value - value2).unwrap()
        );
    }
}
