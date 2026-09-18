// Independent per-decoy error DTMC for threshold acceptance.
dtmc

const int L;
const int T;
const double q;

module iaqd_detection_v2
    sampled : [0..L] init 0;
    errors : [0..L] init 0;

    [sample_particle] sampled<L & errors<L ->
        q : (sampled'=sampled+1) & (errors'=errors+1)
        + (1-q) : (sampled'=sampled+1);
endmodule

label "accept" = sampled=L & errors<=T;
label "reject" = sampled=L & errors>T;

