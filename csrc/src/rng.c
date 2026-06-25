#include "bolr/rng.h"

#include "internal.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

#define BOLR_RNG_SCHEMA_VERSION 1U
#define BOLR_RNG_ALGORITHM_FAMILY 1U
#define BOLR_RNG_ALGORITHM_VERSION 1U
#define BOLR_RNG_PCG_VARIANT 1U
#define BOLR_RNG_ZIGGURAT_LAYERS 128U
#define BOLR_RNG_CHECKPOINT_MAGIC 0x524E4743U
#define BOLR_RNG_CHECKPOINT_VERSION 1U
#define BOLR_RNG_TABLE_HASH 0x8c5330fb82618bc7ULL
#define BOLR_PCG32_MULT 6364136223846793005ULL
#define BOLR_UNIFORM_OPEN01_DENOMINATOR 4294967297.0
#define BOLR_ZIGGURAT_R 3.442619855899
#define BOLR_ZIGGURAT_INV_R 0.29047645161474336

static const uint32_t BOLR_ZIGGURAT_KN[BOLR_RNG_ZIGGURAT_LAYERS] = {
    1991057938u, 0u, 1611602771u, 1826899878u, 1918584482u, 1969227037u, 2001281515u, 2023368125u,
    2039498179u, 2051788381u, 2061460127u, 2069267110u, 2075699398u, 2081089314u, 2085670119u, 2089610331u,
    2093034710u, 2096037586u, 2098691595u, 2101053571u, 2103168620u, 2105072996u, 2106796166u, 2108362327u,
    2109791536u, 2111100552u, 2112303493u, 2113412330u, 2114437283u, 2115387130u, 2116269447u, 2117090813u,
    2117856962u, 2118572919u, 2119243101u, 2119871411u, 2120461303u, 2121015852u, 2121537798u, 2122029592u,
    2122493434u, 2122931299u, 2123344971u, 2123736059u, 2124106020u, 2124456175u, 2124787725u, 2125101763u,
    2125399283u, 2125681194u, 2125948325u, 2126201433u, 2126441213u, 2126668298u, 2126883268u, 2127086657u,
    2127278949u, 2127460589u, 2127631985u, 2127793506u, 2127945490u, 2128088244u, 2128222044u, 2128347141u,
    2128463758u, 2128572095u, 2128672327u, 2128764606u, 2128849065u, 2128925811u, 2128994934u, 2129056501u,
    2129110560u, 2129157136u, 2129196237u, 2129227847u, 2129251929u, 2129268426u, 2129277255u, 2129278312u,
    2129271467u, 2129256561u, 2129233410u, 2129201800u, 2129161480u, 2129112170u, 2129053545u, 2128985244u,
    2128906855u, 2128817916u, 2128717911u, 2128606255u, 2128482298u, 2128345305u, 2128194452u, 2128028813u,
    2127847342u, 2127648860u, 2127432031u, 2127195339u, 2126937058u, 2126655214u, 2126347546u, 2126011445u,
    2125643893u, 2125241376u, 2124799783u, 2124314271u, 2123779094u, 2123187386u, 2122530867u, 2121799464u,
    2120980787u, 2120059418u, 2119015917u, 2117825402u, 2116455471u, 2114863093u, 2112989789u, 2110753906u,
    2108037662u, 2104664315u, 2100355223u, 2094642347u, 2086670106u, 2074676188u, 2054300022u, 2010539237u
};

static const bolr_real BOLR_ZIGGURAT_WN[BOLR_RNG_ZIGGURAT_LAYERS] = {
    1.729040521542798e-09, 1.2680928447002762e-10, 1.6897517773184551e-10, 1.9862688442479051e-10,
    2.2232431792499955e-10, 2.4244936125448931e-10, 2.6016131900632064e-10, 2.7611988711703956e-10,
    2.9073962817715979e-10, 3.0429970414376596e-10, 3.1699795213954273e-10, 3.2898020527113064e-10,
    3.4035738121834064e-10, 3.5121602213664708e-10, 3.616250995056517e-10, 3.7164057634959785e-10,
    3.8130856431105979e-10, 3.9066756809948822e-10, 3.9975011869976912e-10, 4.0858398615984403e-10,
    4.1719309640160654e-10, 4.2559823534592626e-10, 4.3381759739255105e-10, 4.4186721812528858e-10,
    4.4976131962665818e-10, 4.5751258894588287e-10, 4.6513240481400098e-10, 4.7263102384811756e-10,
    4.800177347232567e-10, 4.8730098677987483e-10, 4.9448849805389729e-10, 5.0158734661196158e-10,
    5.0860404824245599e-10, 5.15544622919539e-10, 5.2241465197063155e-10, 5.2921932750063053e-10,
    5.3596349533128897e-10, 5.4265169248206189e-10, 5.4928818003460213e-10, 5.5587697207607733e-10,
    5.6242186129835884e-10, 5.6892644173465501e-10, 5.7539412903756027e-10, 5.8182817863908979e-10,
    5.8823170208121699e-10, 5.9460768176249956e-10, 6.0095898431083022e-10, 6.0728837276278847e-10,
    6.1359851770541355e-10, 6.1989200751559216e-10, 6.2617135781494294e-10, 6.3243902024354019e-10,
    6.3869739064357364e-10, 6.4494881673373833e-10, 6.5119560534646982e-10, 6.5744002929285993e-10,
    6.6368433391398755e-10, 6.6993074337233023e-10, 6.7618146673274439e-10, 6.824387038791137e-10,
    6.8870465131007329e-10, 6.949815078551667e-10, 7.0127148035131547e-10, 7.0757678931855602e-10,
    7.138996746735849e-10, 7.2024240151974857e-10, 7.2660726605270474e-10, 7.329966016220864e-10,
    7.3941278499112283e-10, 7.4585824283835391e-10, 7.5233545854834884e-10, 7.5884697934176525e-10,
    7.6539542379922632e-10, 7.7198348983844004e-10, 7.786139632098381e-10, 7.8528972658289975e-10,
    7.9201376930340978e-10, 7.9878919791135359e-10, 8.0561924752021698e-10, 8.1250729417139681e-10,
    8.1945686829257451e-10, 8.2647166940666245e-10, 8.335555822587845e-10, 8.407126945532991e-10,
    8.4794731652183716e-10, 8.5526400257760939e-10, 8.6266757535193633e-10, 8.7016315245744244e-10,
    8.7775617638032838e-10, 8.8545244797372776e-10, 8.9325816410803695e-10, 9.0117996013566053e-10,
    9.092249579511381e-10, 9.1740082057860052e-10, 9.257158144040126e-10, 9.3417888039884721e-10,
    9.4279971596663144e-10, 9.5158886939988827e-10, 9.6055784938312528e-10, 9.697192525453944e-10,
    9.7908691279089008e-10, 9.8867607706877244e-10, 9.9850361345354251e-10, 1.0085882589914473e-09,
    1.0189509168621382e-09, 1.0296150152006668e-09, 1.0406069436999874e-09, 1.0519565892728039e-09,
    1.0636979991930871e-09, 1.0758702101645819e-09, 1.0885182960607283e-09, 1.1016947078135044e-09,
    1.1154610095597163e-09, 1.1298901613493216e-09, 1.1450695700067237e-09, 1.1611052426022348e-09,
    1.1781275609456131e-09, 1.1962995053850756e-09, 1.2158286983295564e-09, 1.2369856290804966e-09,
    1.2601323300608525e-09, 1.2857696844205153e-09, 1.3146201849677183e-09, 1.3477839562210855e-09,
    1.3870635315067043e-09, 1.435740319181638e-09, 1.5008659030222993e-09, 1.6030947938091123e-09
};

static const bolr_real BOLR_ZIGGURAT_FN[BOLR_RNG_ZIGGURAT_LAYERS] = {
    1.0, 0.96359969312708615, 0.93628268168505957, 0.9130436479717402, 0.8922816507840261, 0.87324304891006954,
    0.85550060786945059, 0.83878360529598961, 0.82290721138140899, 0.80773829468296054, 0.79317701177130506,
    0.7791460859296877, 0.7655841738977045, 0.75244155917461142, 0.73967724367264731, 0.72725691834418482,
    0.7151515074104986, 0.70333609901615812, 0.69178914343667508, 0.68049184099733406, 0.66942766734889037,
    0.65858200005008805, 0.64794182111022247, 0.6374954773350423, 0.62723248524992725, 0.61714337081888093,
    0.60721953662512029, 0.59745315094451668, 0.58783705443470657, 0.57836468111976314, 0.56902999106795094,
    0.55982741270408687, 0.55075179311460454, 0.5417983550254255, 0.53296265938383613, 0.52424057267298407,
    0.51562823824400184, 0.50712205107556896, 0.4987186354709795, 0.49041482528384411, 0.48220764632948521,
    0.47409430069301695, 0.46607215268945612, 0.45813871626787206, 0.45029164368203922, 0.44252871527546844,
    0.43484783024999091, 0.42724699830499607, 0.41972433204957438, 0.412278040102661, 0.40490642080722294,
    0.39760785649387331, 0.39038080823731458, 0.3832238110559012, 0.37613546951056259, 0.36911445366447221,
    0.36215949536931757, 0.35526938484791709, 0.34844296754632659, 0.34167914123155041, 0.33497685331358917,
    0.3283350983728503, 0.32175291587598492, 0.31522938806501088, 0.30876363800618112, 0.30235482778648354,
    0.29600215684693298, 0.28970486044295984, 0.28346220822323298, 0.27727350291918812, 0.27113807913838461,
    0.26505530225558921, 0.25902456739620483, 0.25304529850732577, 0.24711694751232141, 0.24123899354543982,
    0.23541094226347908, 0.22963232523211613, 0.22390269938500842, 0.2182216465543054, 0.2125887730717303,
    0.20700370943992652, 0.20146611007431367, 0.19597565311627774, 0.19053204031913715, 0.18513499700899219,
    0.17978427212329545, 0.1744796383307895, 0.169220892237365, 0.16400785468342038, 0.1588403711394793,
    0.15371831220818166, 0.14864157424234226, 0.14361008009062776, 0.1386237799845946, 0.13368265258343937,
    0.12878670619594321, 0.12393598020286782, 0.11913054670765083, 0.11437051244886601, 0.10965602101484027,
    0.10498725540942132, 0.10036444102865587, 0.095787849121731439, 0.091257800826830257, 0.086774671894780178,
    0.082338898242235656, 0.077950982513973394, 0.073611501884113403, 0.069321117393577908, 0.065080585213068073,
    0.060890770348040406, 0.056752663481049848, 0.052667401903051012, 0.048636295859867805, 0.044660862200491425,
    0.040742868074444175, 0.036884388786656203, 0.033087886146225751, 0.02935631744000685, 0.025693291935934271,
    0.022103304615927098, 0.018592102737011288, 0.015167298010546568, 0.011839478657884862, 0.0086244844128598851,
    0.0055489952207713449, 0.0026696290838809228
};

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint64_t state;
    uint64_t increment;
    bolr_rng_metadata metadata;
} bolr_rng_checkpoint_wire;

static void rng_assign_metadata(struct bolr_rng *rng) {
    rng->schema_version = BOLR_RNG_SCHEMA_VERSION;
    rng->algorithm_family = BOLR_RNG_ALGORITHM_FAMILY;
    rng->algorithm_version = BOLR_RNG_ALGORITHM_VERSION;
    rng->pcg_variant = BOLR_RNG_PCG_VARIANT;
    rng->ziggurat_layers = BOLR_RNG_ZIGGURAT_LAYERS;
    rng->table_hash = BOLR_RNG_TABLE_HASH;
}

static void rng_copy_metadata_fields(const struct bolr_rng *rng, bolr_rng_metadata *out_metadata) {
    out_metadata->schema_version = rng->schema_version;
    out_metadata->algorithm_family = rng->algorithm_family;
    out_metadata->algorithm_version = rng->algorithm_version;
    out_metadata->pcg_variant = rng->pcg_variant;
    out_metadata->ziggurat_layers = rng->ziggurat_layers;
    out_metadata->table_hash = rng->table_hash;
    out_metadata->seed = rng->seed.seed;
    out_metadata->stream = rng->seed.stream;
    out_metadata->u32_draw_count = rng->u32_draw_count;
    out_metadata->uniform_draw_count = rng->uniform_draw_count;
    out_metadata->normal_draw_count = rng->normal_draw_count;
}

static bolr_status rng_validate_metadata(const bolr_rng_metadata *metadata) {
    if (metadata == NULL) return BOLR_INVALID_ARGUMENT;
    if (metadata->schema_version != BOLR_RNG_SCHEMA_VERSION) return BOLR_INCOMPATIBLE_CHECKPOINT;
    if (metadata->algorithm_family != BOLR_RNG_ALGORITHM_FAMILY) return BOLR_INCOMPATIBLE_CHECKPOINT;
    if (metadata->algorithm_version != BOLR_RNG_ALGORITHM_VERSION) return BOLR_INCOMPATIBLE_CHECKPOINT;
    if (metadata->pcg_variant != BOLR_RNG_PCG_VARIANT) return BOLR_INCOMPATIBLE_CHECKPOINT;
    if (metadata->ziggurat_layers != BOLR_RNG_ZIGGURAT_LAYERS) return BOLR_INCOMPATIBLE_CHECKPOINT;
    if (metadata->table_hash != BOLR_RNG_TABLE_HASH) return BOLR_INCOMPATIBLE_CHECKPOINT;
    return BOLR_OK;
}

static uint32_t pcg32_next(struct bolr_rng *rng) {
    uint64_t oldstate = rng->state;
    uint32_t xorshifted;
    uint32_t rot;
    rng->state = oldstate * BOLR_PCG32_MULT + rng->increment;
    rng->u32_draw_count += 1U;
    xorshifted = (uint32_t) (((oldstate >> 18U) ^ oldstate) >> 27U);
    rot = (uint32_t) (oldstate >> 59U);
    return (xorshifted >> rot) | (xorshifted << ((uint32_t) (-(int32_t) rot) & 31U));
}

static bolr_status rng_seed_handle(struct bolr_rng *rng, bolr_rng_seed seed) {
    if ((rng == NULL) || ((seed.stream << 1U) == UINT64_MAX)) return BOLR_INVALID_ARGUMENT;
    rng->state = 0ULL;
    rng->increment = (seed.stream << 1U) | 1ULL;
    rng->seed = seed;
    rng->u32_draw_count = 0ULL;
    rng->uniform_draw_count = 0ULL;
    rng->normal_draw_count = 0ULL;
    rng_assign_metadata(rng);
    (void) pcg32_next(rng);
    rng->state += seed.seed;
    (void) pcg32_next(rng);
    return BOLR_OK;
}

static bolr_real sample_tail(struct bolr_rng *rng) {
    bolr_real x;
    bolr_real y;
    do {
        x = -log((((bolr_real) pcg32_next(rng)) + 1.0) / BOLR_UNIFORM_OPEN01_DENOMINATOR) * BOLR_ZIGGURAT_INV_R;
        y = -log((((bolr_real) pcg32_next(rng)) + 1.0) / BOLR_UNIFORM_OPEN01_DENOMINATOR);
    } while ((y + y) < (x * x));
    return BOLR_ZIGGURAT_R + x;
}

bolr_status bolr_rng_create(bolr_rng_seed seed, const bolr_allocator *allocator, bolr_rng **out_rng) {
    const bolr_allocator *active = (allocator == NULL) ? bolr_default_allocator() : allocator;
    struct bolr_rng *rng;
    if (out_rng == NULL) return BOLR_INVALID_ARGUMENT;
    *out_rng = NULL;
    rng = (struct bolr_rng *) bolr_allocator_calloc(active, 1U, sizeof(*rng));
    if (rng == NULL) return BOLR_ALLOCATION_FAILED;
    rng->allocator = active;
    if (rng_seed_handle(rng, seed) != BOLR_OK) {
        bolr_allocator_free(active, rng);
        return BOLR_INVALID_ARGUMENT;
    }
    *out_rng = rng;
    return BOLR_OK;
}

void bolr_rng_destroy(bolr_rng *rng) {
    if (rng == NULL) return;
    bolr_allocator_free(rng->allocator, rng);
}

bolr_status bolr_rng_clone(const bolr_rng *source, const bolr_allocator *allocator, bolr_rng **out_clone) {
    const bolr_allocator *active = (allocator == NULL) ? bolr_default_allocator() : allocator;
    struct bolr_rng *clone;
    if ((source == NULL) || (out_clone == NULL)) return BOLR_INVALID_ARGUMENT;
    *out_clone = NULL;
    clone = (struct bolr_rng *) bolr_allocator_malloc(active, sizeof(*clone));
    if (clone == NULL) return BOLR_ALLOCATION_FAILED;
    memcpy(clone, source, sizeof(*clone));
    clone->allocator = active;
    *out_clone = clone;
    return BOLR_OK;
}

bolr_status bolr_rng_u32(bolr_rng *rng, uint32_t *out) {
    if ((rng == NULL) || (out == NULL)) return BOLR_INVALID_ARGUMENT;
    *out = pcg32_next(rng);
    return BOLR_OK;
}

bolr_status bolr_rng_uniform_open01(bolr_rng *rng, bolr_real *out) {
    uint32_t value;
    if ((rng == NULL) || (out == NULL)) return BOLR_INVALID_ARGUMENT;
    value = pcg32_next(rng);
    rng->uniform_draw_count += 1U;
    *out = (((bolr_real) value) + 1.0) / BOLR_UNIFORM_OPEN01_DENOMINATOR;
    return BOLR_OK;
}

bolr_status bolr_rng_fill_uniform_open01(bolr_rng *rng, bolr_vector_view output) {
    bolr_index i;
    if ((rng == NULL) || (bolr_mutable_vector_view_validate(output) != BOLR_OK)) return BOLR_INVALID_ARGUMENT;
    for (i = 0; i < output.length; ++i) {
        bolr_status status = bolr_rng_uniform_open01(rng, &output.data[i * output.stride]);
        if (status != BOLR_OK) return status;
    }
    return BOLR_OK;
}

bolr_status bolr_rng_standard_normal(bolr_rng *rng, bolr_real *out) {
    if ((rng == NULL) || (out == NULL)) return BOLR_INVALID_ARGUMENT;
    for (;;) {
        uint32_t u = pcg32_next(rng);
        uint32_t iz = u & (BOLR_RNG_ZIGGURAT_LAYERS - 1U);
        uint32_t uabs = u & 0x7fffffffU;
        bolr_real x = ((bolr_real) uabs) * BOLR_ZIGGURAT_WN[iz];
        if (uabs < BOLR_ZIGGURAT_KN[iz]) {
            rng->normal_draw_count += 1U;
            *out = (u & 0x80000000U) ? -x : x;
            return BOLR_OK;
        }
        if (iz == 0U) {
            x = sample_tail(rng);
            rng->normal_draw_count += 1U;
            *out = (u & 0x80000000U) ? -x : x;
            return BOLR_OK;
        }
        {
            bolr_real y;
            bolr_real uniform;
            uniform = (((bolr_real) pcg32_next(rng)) + 1.0) / BOLR_UNIFORM_OPEN01_DENOMINATOR;
            y = BOLR_ZIGGURAT_FN[iz] + uniform * (BOLR_ZIGGURAT_FN[iz - 1U] - BOLR_ZIGGURAT_FN[iz]);
            if (y < exp(-0.5 * x * x)) {
                rng->normal_draw_count += 1U;
                *out = (u & 0x80000000U) ? -x : x;
                return BOLR_OK;
            }
        }
    }
}

bolr_status bolr_rng_fill_standard_normal(bolr_rng *rng, bolr_vector_view output) {
    bolr_index i;
    if ((rng == NULL) || (bolr_mutable_vector_view_validate(output) != BOLR_OK)) return BOLR_INVALID_ARGUMENT;
    for (i = 0; i < output.length; ++i) {
        bolr_status status = bolr_rng_standard_normal(rng, &output.data[i * output.stride]);
        if (status != BOLR_OK) return status;
    }
    return BOLR_OK;
}

bolr_status bolr_rng_metadata_copy(const bolr_rng *rng, bolr_rng_metadata *out_metadata) {
    if ((rng == NULL) || (out_metadata == NULL)) return BOLR_INVALID_ARGUMENT;
    rng_copy_metadata_fields(rng, out_metadata);
    return BOLR_OK;
}

bolr_status bolr_rng_export(const bolr_rng *rng, const bolr_allocator *allocator, bolr_rng_checkpoint **out_checkpoint) {
    const bolr_allocator *active = (allocator == NULL) ? bolr_default_allocator() : allocator;
    struct bolr_rng_checkpoint *checkpoint;
    if ((rng == NULL) || (out_checkpoint == NULL)) return BOLR_INVALID_ARGUMENT;
    *out_checkpoint = NULL;
    checkpoint = (struct bolr_rng_checkpoint *) bolr_allocator_calloc(active, 1U, sizeof(*checkpoint));
    if (checkpoint == NULL) return BOLR_ALLOCATION_FAILED;
    checkpoint->allocator = active;
    checkpoint->state = rng->state;
    checkpoint->increment = rng->increment;
    rng_copy_metadata_fields(rng, &checkpoint->metadata);
    *out_checkpoint = checkpoint;
    return BOLR_OK;
}

bolr_status bolr_rng_import(const bolr_rng_checkpoint *checkpoint, const bolr_allocator *allocator, bolr_rng **out_rng) {
    const bolr_allocator *active = (allocator == NULL) ? bolr_default_allocator() : allocator;
    struct bolr_rng *rng;
    bolr_status status;
    if ((checkpoint == NULL) || (out_rng == NULL)) return BOLR_INVALID_ARGUMENT;
    *out_rng = NULL;
    status = rng_validate_metadata(&checkpoint->metadata);
    if (status != BOLR_OK) return status;
    if ((checkpoint->increment & 1ULL) == 0ULL) return BOLR_INCOMPATIBLE_CHECKPOINT;
    rng = (struct bolr_rng *) bolr_allocator_calloc(active, 1U, sizeof(*rng));
    if (rng == NULL) return BOLR_ALLOCATION_FAILED;
    rng->allocator = active;
    rng->state = checkpoint->state;
    rng->increment = checkpoint->increment;
    rng->seed.seed = checkpoint->metadata.seed;
    rng->seed.stream = checkpoint->metadata.stream;
    rng->u32_draw_count = checkpoint->metadata.u32_draw_count;
    rng->uniform_draw_count = checkpoint->metadata.uniform_draw_count;
    rng->normal_draw_count = checkpoint->metadata.normal_draw_count;
    rng_assign_metadata(rng);
    *out_rng = rng;
    return BOLR_OK;
}

void bolr_rng_checkpoint_destroy(bolr_rng_checkpoint *checkpoint) {
    if (checkpoint == NULL) return;
    bolr_allocator_free(checkpoint->allocator, checkpoint);
}

bolr_status bolr_rng_checkpoint_metadata_copy(const bolr_rng_checkpoint *checkpoint, bolr_rng_metadata *out_metadata) {
    if ((checkpoint == NULL) || (out_metadata == NULL)) return BOLR_INVALID_ARGUMENT;
    *out_metadata = checkpoint->metadata;
    return BOLR_OK;
}

bolr_status bolr_rng_checkpoint_encoded_size(const bolr_rng_checkpoint *checkpoint, size_t *out_size) {
    if ((checkpoint == NULL) || (out_size == NULL)) return BOLR_INVALID_ARGUMENT;
    *out_size = sizeof(bolr_rng_checkpoint_wire);
    return BOLR_OK;
}

bolr_status bolr_rng_checkpoint_encode(const bolr_rng_checkpoint *checkpoint, void *output, size_t output_size, size_t *out_written) {
    bolr_rng_checkpoint_wire wire;
    if ((checkpoint == NULL) || (output == NULL)) return BOLR_INVALID_ARGUMENT;
    if (output_size < sizeof(wire)) return BOLR_INVALID_SHAPE;
    wire.magic = BOLR_RNG_CHECKPOINT_MAGIC;
    wire.version = BOLR_RNG_CHECKPOINT_VERSION;
    wire.state = checkpoint->state;
    wire.increment = checkpoint->increment;
    wire.metadata = checkpoint->metadata;
    memcpy(output, &wire, sizeof(wire));
    if (out_written != NULL) *out_written = sizeof(wire);
    return BOLR_OK;
}

bolr_status bolr_rng_checkpoint_decode(const void *data, size_t data_size, const bolr_allocator *allocator, bolr_rng_checkpoint **out_checkpoint) {
    const bolr_allocator *active = (allocator == NULL) ? bolr_default_allocator() : allocator;
    const bolr_rng_checkpoint_wire *wire;
    struct bolr_rng_checkpoint *checkpoint;
    if ((data == NULL) || (out_checkpoint == NULL)) return BOLR_INVALID_ARGUMENT;
    *out_checkpoint = NULL;
    if (data_size != sizeof(*wire)) return BOLR_INCOMPATIBLE_CHECKPOINT;
    wire = (const bolr_rng_checkpoint_wire *) data;
    if ((wire->magic != BOLR_RNG_CHECKPOINT_MAGIC) || (wire->version != BOLR_RNG_CHECKPOINT_VERSION)) return BOLR_INCOMPATIBLE_CHECKPOINT;
    if ((wire->increment & 1ULL) == 0ULL) return BOLR_INCOMPATIBLE_CHECKPOINT;
    if (rng_validate_metadata(&wire->metadata) != BOLR_OK) return BOLR_INCOMPATIBLE_CHECKPOINT;
    checkpoint = (struct bolr_rng_checkpoint *) bolr_allocator_calloc(active, 1U, sizeof(*checkpoint));
    if (checkpoint == NULL) return BOLR_ALLOCATION_FAILED;
    checkpoint->allocator = active;
    checkpoint->state = wire->state;
    checkpoint->increment = wire->increment;
    checkpoint->metadata = wire->metadata;
    *out_checkpoint = checkpoint;
    return BOLR_OK;
}
