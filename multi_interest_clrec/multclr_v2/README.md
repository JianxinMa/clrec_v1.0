A refactored version of multclr_v1, but may contain some weird code that I once used for some offline experiments.

Warning: The code may not run on your system because it uses our internal, heavily modified version of tensorflow. I
have removed some code that relies on our internal infrastructure, but have no time to make it compatible with the
public implementation of tensorflow yet.

Note: It contains our multi-interest encoder, but not exactly the same one as the multi-queue model described in the
paper. In fact, it is a predecessor of the multi-queue model. This is because our implementation of the multi-queue
model is too deeply coupled with our internal tools and requires significantly more efforts to make it readable enough
for releasing.
