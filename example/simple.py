#!/usr/bin/env python

from collections import namedtuple
import numpy as np
import galsim
from tqdm import tqdm
from matplotlib import pyplot

class Transform(namedtuple("Jacobian", ["dudx", "dudy", "dvdx", "dvdy", "du", "dv"])):

    def __call__(self, x, y=None):
        if y is None:
            if isinstance(x, galsim.PositionD):
                return galsim.PositionD(*self(x.x, x.y))
            x, y = x
        return (self.dudx*x + self.dudy*y + self.du, self.dvdx*x + self.dvdy*y + self.dv)

    @property
    def jacobian(self):
        return self[:4]

    @property
    def offset(self):
        return galsim.PositionD(self.du, self.dv)

    @property
    def det(self):
        return self.dudx*self.dvdy - self.dudy*self.dvdx

    @property
    def inverse(self):
        det = self.det
        return Transform(self.dvdy/det, -self.dudy/det, -self.dvdx/det, self.dudx/det,
                         (self.dudy*self.dv - self.dvdy*self.du)/det,
                         (self.dvdx*self.du - self.dudx*self.dv)/det)

    @staticmethod
    def test():
        a = Transform(*np.random.randn(6))
        b = a.inverse
        p = np.random.randn(2)
        assert np.allclose(b(a(p)), p)
        assert np.allclose(a(b(p)), p)


class Frame:

    def __init__(self, psf, transform, bounds):
        self.psf = psf
        self.transform = transform
        self.image = galsim.Image(bounds, dtype=np.float64, scale=1)

    @property
    def bounds(self):
        return self.image.bounds

    def draw(self, scene):
        inv = self.transform.inverse
        s = galsim.Convolve(self.psf, galsim.Transform(scene, inv.jacobian, inv.offset))
        s.drawImage(image=self.image, add_to_image=True, method='no_pixel', offset=-self.image.center)

    def psf_at(self, coadd_position):
        inv = self.transform.inverse
        image = galsim.Image(self.bounds, dtype=np.float64, scale=1)
        frame_position = inv(coadd_position)
        galsim.Transform(self.psf, offset=-frame_position).drawImage(image=image, method='no_pixel', offset=-image.center)
        image.array[:, :] /= self.transform.det
        return image

    def show(self):
        pyplot.imshow(self.image.array, origin='lower', interpolation='nearest',
                      extent=(self.bounds.xmin-0.5, self.bounds.xmax+0.5,
                              self.bounds.ymin-0.5, self.bounds.ymax+0.5))


def run():
    f1 = Frame(
        psf=galsim.Convolve(galsim.Gaussian(fwhm=4, flux=1), galsim.Pixel(scale=1)),
        transform=Transform(1.0, 0.0, 0.0, 1.0, 0.0, 5.0),
        bounds=galsim.BoundsI(-25, 35, -20, 20)
    )
    f2 = Frame(
        psf=galsim.Convolve(galsim.Gaussian(fwhm=5, flux=1), galsim.Pixel(scale=1)),
        transform=Transform(0.5, 0.0, 0.0, 0.5, -2.5, 0.0),
        bounds=galsim.BoundsI(-20, 20, -35, 25)
    )
    position = galsim.PositionD(-3, 4)
    scene = galsim.Transform(galsim.DeltaFunction(flux=1), offset=position)
    f1.draw(scene)
    f2.draw(scene)
    #Phi_1 = np.zeros((21, 23, 21, 23), dtype=np.float64)
    #Phi_2 = np.zeros((21, 23, 21, 23), dtype=np.float64)
    extra_1 = np.zeros((21, 23, 21, 23), dtype=np.complex64)
    extra_2 = np.zeros((21, 23, 21, 23), dtype=np.complex64)
    x1, y1 = np.meshgrid(np.arange(f1.bounds.xmin, f1.bounds.xmax + 1),
                         np.arange(f1.bounds.ymin, f1.bounds.ymax + 1))
    x2, y2 = np.meshgrid(np.arange(f2.bounds.xmin, f2.bounds.xmax + 1),
                         np.arange(f2.bounds.ymin, f2.bounds.ymax + 1))
    eta1, zeta1 = f1.transform(x1, y1)
    eta2, zeta2 = f2.transform(x2, y2)
    k1 = -2*np.pi/f1.bounds.area()
    k2 = -2*np.pi/f2.bounds.area()
    with tqdm(total=extra_1.size) as progress:
        for i0, mu_x in enumerate(range(-10, 11)):
            for i1, mu_y in enumerate(range(-11, 12)):
                #phi_mu_1 = f1.psf_at(galsim.PositionD(mu_x, mu_y))
                #phi_mu_2 = f2.psf_at(galsim.PositionD(mu_x, mu_y))
                for i2, nu_x in enumerate(range(-10, 11)):
                    for i3, nu_y in enumerate(range(-11, 12)):
                        #phi_nu_1 = f1.psf_at(galsim.PositionD(nu_x, nu_y))
                        #phi_nu_2 = f2.psf_at(galsim.PositionD(nu_x, nu_y))
                        #Phi_1[i0, i1, i2, i3] = (phi_mu_1.array*phi_nu_1.array).sum()
                        #Phi_2[i0, i1, i2, i3] = (phi_mu_2.array*phi_nu_2.array).sum()
                        arg1 = k1*((mu_x - nu_x)*eta1 + (mu_y - nu_y)*zeta1)
                        arg2 = k2*((mu_x - nu_x)*eta2 + (mu_y - nu_y)*zeta2)
                        extra_1[i0, i1, i2, i3] = np.sum(np.cos(arg1) + 1j*np.sin(arg1))
                        extra_2[i0, i1, i2, i3] = np.sum(np.cos(arg2) + 1j*np.sin(arg2))
                        progress.update(1)
    pyplot.matshow(extra_1.reshape(21*23, 21*23).real)
    pyplot.matshow(extra_2.reshape(21*23, 21*23).imag)
    pyplot.show()

if __name__ == "__main__":
    Transform.test()
    run()
