//! Output types and functions for the [`TextEmbedding`] model.
//!
use crate::{
    common::{normalize, Embedding},
    output::{OutputKey, OutputPrecedence, SingleBatchOutput},
    pooling::Pooling,
};

#[cfg(doc)]
use super::TextEmbedding;

/// The default output precedence for the TextEmbedding model.
pub const OUTPUT_TYPE_PRECEDENCE: &[OutputKey] = &[
    OutputKey::OnlyOne,
    OutputKey::ByName("text_embeds"),
    OutputKey::ByName("last_hidden_state"),
    OutputKey::ByName("sentence_embedding"),
    // Better not to expose this unless the user explicitly asks for it.
    // OutputKey::ByName("token_embeddings"),
];

/// Generates thea default array transformer for the [`TextEmbedding`] model using the
/// provided output precedence.
///
// TODO (denwong47): now that pooling is done in SingleBatchOutput, it is possible that
// all the models will use this same generic transformer. Move this into SingleBatchOutput?
#[allow(unused_variables)]
pub fn transformer_with_precedence(
    output_precedence: impl OutputPrecedence,
    pooling: Option<Pooling>,
) -> impl Fn(&[SingleBatchOutput]) -> anyhow::Result<Vec<Embedding>> {
    move |batches| {
        // Not using `par_iter` here: the operations here is probably not
        // computationally expensive enough to warrant spinning up costs of the threads.
        batches
            .iter()
            .map(|batch| {
                batch
                    .select_and_pool_output(&output_precedence, pooling.clone())
                    .map(|array| {
                        array
                            .rows()
                            .into_iter()
                            .map(|row| normalize(row.as_slice().unwrap()))
                            .collect::<Vec<Embedding>>()
                    })
            })
            .try_fold(Vec::new(), |mut acc, res| {
                acc.extend(res?);
                Ok(acc)
            })
    }
}

pub fn transformer_flat(
    output_precedence: impl OutputPrecedence,
    pooling: Option<Pooling>,
) -> impl Fn(&[SingleBatchOutput]) -> anyhow::Result<(Vec<f32>, usize, usize)> {
    move |batches| {
        let mut all = Vec::new();
        let mut rows = 0;
        let mut cols = 0;

        for batch in batches {
            let array = batch.select_and_pool_output(&output_precedence, pooling.clone())?;
            let (r, c) = array.dim();
            if cols == 0 {
                cols = c;
            } else {
                debug_assert_eq!(cols, c, "inconsistent embedding dimensions");
            }
            rows += r;

            let (mut vec, offset) = array.into_raw_vec_and_offset();
            debug_assert_eq!(offset, Some(0), "expected contiguous array with offset 0");

            all.append(&mut vec);
        }

        Ok((all, rows, cols))
    }
}
